import pytest
import numpy as np
import scipy.sparse.linalg as spla
import sympy as sp

# Assume these are imported from your project structure
from pycutfem.core.mesh import Mesh
from pycutfem.assembly.dg_global import assemble_dg
from pycutfem.utils.meshgen import structured_quad, structured_triangles
from pycutfem.fem.reference import get_reference
from pycutfem.fem import transform
from pycutfem.integration import volume

# --- Define analytical solutions and source terms ---

# Scalar case
x_s, y_s = sp.symbols("x y")
u_s_sym = x_s**3 * y_s**2 + sp.sin(sp.pi * y_s)
f_s_sym = -sp.diff(u_s_sym, x_s, 2) - sp.diff(u_s_sym, y_s, 2)
u_exact_scalar = sp.lambdify((x_s, y_s), u_s_sym, "numpy")
source_scalar = sp.lambdify((x_s, y_s), f_s_sym, "numpy")

# Vector case
u_v_sym = sp.Matrix([x_s**2 * y_s, x_s * y_s**2])
f_v_sym = sp.Matrix([-sp.diff(c, x_s, 2) - sp.diff(c, y_s, 2) for c in u_v_sym])
u_exact_vector = sp.lambdify((x_s, y_s), u_v_sym, "numpy")
source_vector = sp.lambdify((x_s, y_s), f_v_sym, "numpy")


def test_sipg_q1_poisson():
    """
    Tests the DG solver for a Poisson problem with Q1 elements.
    """
    # 1. Setup
    nodes, elems, _, corners = structured_quad(1, 1, nx=8, ny=8, poly_order=1)
    # FIX: Added edges_connectivity=None to match constructor
    mesh = Mesh(nodes, element_connectivity=elems, edges_connectivity=None, elements_corner_nodes=corners, element_type="quad", poly_order=1)

    # 2. Assemble
    K, F = assemble_dg(mesh, alpha=20.0, symmetry=1,
                       dirichlet=lambda x, y: u_exact_scalar(x, y),
                       rhs=lambda x, y: source_scalar(x, y))

    # 3. Solve
    uh = spla.spsolve(K, F)

    # 4. Compute L2 error
    l2_error = compute_l2_error(mesh, uh, u_exact_scalar, n_comp=1)
    assert l2_error < 0.1, f"Q1 L2 error ({l2_error}) is too high."


def test_sipg_q2_poisson():
    """
    Tests the DG solver for a Poisson problem with Q2 elements.
    """
    # 1. Setup
    nodes, elems, _, corners = structured_quad(1, 1, nx=8, ny=8, poly_order=2)
    # FIX: Added edges_connectivity=None to match constructor
    mesh = Mesh(nodes, element_connectivity=elems, edges_connectivity=None, elements_corner_nodes=corners, element_type="quad", poly_order=2)

    # 2. Assemble
    K, F = assemble_dg(mesh, alpha=50.0, symmetry=1,
                       dirichlet=lambda x, y: u_exact_scalar(x, y),
                       rhs=lambda x, y: source_scalar(x, y))

    # 3. Solve
    uh = spla.spsolve(K, F)

    # 4. Compute L2 error
    l2_error = compute_l2_error(mesh, uh, u_exact_scalar, n_comp=1)
    # Higher order should have lower error
    assert l2_error < 0.01, f"Q2 L2 error ({l2_error}) is too high."


# def test_sipg_p2_poisson():
#     """
#     Tests the DG solver for a Poisson problem with P2 (triangular) elements.
#     """
#     # 1. Setup
#     poly_order = 1
#     ncomp = 1
#     nodes, elems, _, corners = structured_triangles(1, 1, nx_quads=8, ny_quads=8, poly_order=poly_order)
#     # FIX: Added edges_connectivity=None to match constructor
#     mesh = Mesh(nodes, element_connectivity=elems, edges_connectivity=None, elements_corner_nodes=corners, 
#                 element_type="tri", poly_order=poly_order)
#     # print(f'elements: {mesh.elements_list}')
#     # print(f'edges: {mesh.edges_list}')
#     # print(f'nodes: {mesh.nodes_list}')
#     # 2. Assemble
#     # FIX: Increased penalty parameter alpha for P2 elements as they can be more
#     # sensitive, which can improve stability and accuracy.
#     K, F = assemble_dg(mesh, alpha=20.0, symmetry=1,
#                        dirichlet=lambda x, y: u_exact_scalar(x, y),
#                        rhs=lambda x, y: source_scalar(x, y),
#                        n_comp=ncomp)

#     # 3. Solve
#     uh = spla.spsolve(K, F)

#     # 4. Compute L2 error
#     l2_error = compute_l2_error(mesh, uh, u_exact_scalar, n_comp=1)
#     assert l2_error < 0.01, f"P2 L2 error ({l2_error}) is too high."


def test_sipg_vector_q2_poisson():
    """
    Tests the DG solver for a 2-component vector Poisson problem with Q2 elements.
    """
    # 1. Setup
    nodes, elems, _, corners = structured_quad(1, 1, nx=8, ny=8, poly_order=2)
    # FIX: Added edges_connectivity=None to match constructor
    mesh = Mesh(nodes, element_connectivity=elems, edges_connectivity=None, elements_corner_nodes=corners, element_type="quad", poly_order=2)

    # 2. Assemble
    K, F = assemble_dg(mesh, n_comp=2, alpha=50.0, symmetry=1,
                       dirichlet=lambda x, y: u_exact_vector(x, y),
                       rhs=lambda x, y: source_vector(x, y))

    # 3. Solve
    uh = spla.spsolve(K, F)

    # 4. Compute L2 error
    l2_error = compute_l2_error(mesh, uh, u_exact_vector, n_comp=2)
    assert l2_error < 0.01, f"Vector Q2 L2 error ({l2_error}) is too high."


# --- Helper function for error computation ---

def compute_l2_error(mesh, uh, u_exact_func, *, n_comp):
    """Computes the L2 error for a scalar or vector solution."""
    total_error_sq = 0.0
    total_area = 0.0
    
    ref = get_reference(mesh.element_type, mesh.poly_order)
    pts, wts = volume(mesh.element_type, 2 * mesh.poly_order + 2) # Increased quadrature
    n_loc = len(ref.shape(0, 0))
    n_eldof = n_loc * n_comp

    for eid, elem in enumerate(mesh.elements_list):
        dofs = np.arange(n_eldof) + eid * n_eldof
        uh_element = uh[dofs]
        
        elem_error_sq = 0.0
        for (xi, eta), w in zip(pts, wts):
            N = ref.shape(xi, eta)
            J = transform.jacobian(mesh, eid, (xi, eta))
            detJ = abs(np.linalg.det(J))
            x_phys = transform.x_mapping(mesh, eid, (xi, eta))
            
            u_exact_at_pt = np.array(u_exact_func(*x_phys)).flatten()

            # Handle scalar vs vector
            uh_at_pt = np.zeros(n_comp)
            for c in range(n_comp):
                uh_local_comp = uh_element[c * n_loc:(c + 1) * n_loc]
                uh_at_pt[c] = N @ uh_local_comp

            error_vec = uh_at_pt - u_exact_at_pt
            elem_error_sq += w * detJ * (error_vec @ error_vec)
            
        total_error_sq += elem_error_sq
        total_area += mesh.areas()[eid]

    return np.sqrt(total_error_sq / total_area)
