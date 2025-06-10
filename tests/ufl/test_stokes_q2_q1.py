import pytest
import numpy as np
import scipy.sparse.linalg as spla

# --- Core imports from the refactored library ---
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad

# --- UFL-like imports ---
from ufl.expressions import TrialFunction, TestFunction, grad, inner, dot, div, Constant
from ufl.measures import dx
from ufl.forms import BoundaryCondition, assemble_form

def test_stokes_lid_driven_cavity():
    """
    Tests the solver for the Stokes system using Q2-Q1 Taylor-Hood elements.
    This solves the lid-driven cavity problem.

    -∇²u + ∇p = f  (Momentum)
         ∇⋅u = 0    (Continuity)
    """
    # 1. Setup Meshes and DofHandler for Q2-Q1 elements
    nodes_q2, elems_q2, _, corners_q2 = structured_quad(1, 1, nx=4, ny=4, poly_order=2)
    mesh_q2 = Mesh(nodes=nodes_q2, element_connectivity=elems_q2,
                   elements_corner_nodes=corners_q2, element_type="quad", poly_order=2)

    nodes_q1, elems_q1, _, _ = structured_quad(1, 1, nx=4, ny=4, poly_order=1)
    mesh_q1 = Mesh(nodes=nodes_q1, element_connectivity=elems_q1,
                   elements_corner_nodes=elems_q1, element_type="quad", poly_order=1)

    fe_map = {
        'ux': mesh_q2, 'uy': mesh_q2, 'p': mesh_q1
    }
    dof_handler = DofHandler(fe_map, method='cg')

    # 2. Define Trial and Test Functions for the system
    ux, uy = TrialFunction('ux'), TrialFunction('uy')
    vx, vy = TestFunction('ux'), TestFunction('uy')
    p, q = TrialFunction('p'), TestFunction('p')

    # 3. Define the Weak Form for the Stokes System component-wise
    mom_x_form = (inner(grad(ux), grad(vx)) - p * grad(vx)[0]) * dx()
    eq_mom_x = mom_x_form == Constant(0) * vx * dx()

    mom_y_form = (inner(grad(uy), grad(vy)) - p * grad(vy)[1]) * dx()
    eq_mom_y = mom_y_form == Constant(0) * vy * dx()

    cont_form = q * (grad(ux)[0] + grad(uy)[1]) * dx()
    eq_cont = cont_form == Constant(0) * q * dx()
    
    system = [eq_mom_x, eq_mom_y, eq_cont]

    # 4. Define Boundary Conditions for the Lid-Driven Cavity
    bc_tags = {
        'bottom_wall': lambda x, y: np.isclose(y, 0),
        'left_wall':   lambda x, y: np.isclose(x, 0),
        'right_wall':  lambda x, y: np.isclose(x, 1),
        'top_lid':     lambda x, y: np.isclose(y, 1)
    }
    mesh_q2.tag_boundary_edges(bc_tags)
    
    # Define a special tag for the pressure pinning point.
    # We tag a single node. The DofHandler will now find this.
    mesh_q1.nodes_list[0].tag = 'pressure_pin_point'

    bcs = [
        # Velocity BCs
        BoundaryCondition('ux', 'dirichlet', 'left_wall',   lambda x, y: 0.0),
        BoundaryCondition('uy', 'dirichlet', 'left_wall',   lambda x, y: 0.0),
        BoundaryCondition('ux', 'dirichlet', 'right_wall',  lambda x, y: 0.0),
        BoundaryCondition('uy', 'dirichlet', 'right_wall',  lambda x, y: 0.0),
        BoundaryCondition('ux', 'dirichlet', 'bottom_wall', lambda x, y: 0.0),
        BoundaryCondition('uy', 'dirichlet', 'bottom_wall', lambda x, y: 0.0),
        BoundaryCondition('ux', 'dirichlet', 'top_lid',     lambda x, y: 1.0),
        BoundaryCondition('uy', 'dirichlet', 'top_lid',     lambda x, y: 0.0),
        
        # Add an explicit BC to pin the pressure at a single point to make the system non-singular.
        BoundaryCondition('p', 'dirichlet', 'pressure_pin_point', lambda x, y: 0.0)
    ]

    # 5. Assemble and Solve the System
    K, F = assemble_form(system, dof_handler=dof_handler, bcs=bcs, quad_order=5)
    
    # --- DEBUG: Check matrix rank ---
    # A singular matrix will have rank < number of rows/columns.
    # We check the rank of the matrix *after* BCs are applied.
    # The rank should be equal to the total number of DOFs.
    matrix_rank = np.linalg.matrix_rank(K.toarray())
    print(f"\nMatrix Info: Shape={K.shape}, Rank={matrix_rank}")
    assert matrix_rank == K.shape[0], f"Matrix is rank-deficient! Rank is {matrix_rank}, but shape is {K.shape}. The system is singular."

    solution_vec = spla.spsolve(K, F)

    # 6. Verify Solution (Simple Checks)
    top_center_node_id = -1
    for node in mesh_q2.nodes_list:
        if np.isclose(node.x, 0.5) and np.isclose(node.y, 1.0):
            top_center_node_id = node.id
            break

    ux_top_dof = dof_handler.dof_map['ux'][top_center_node_id]
    print(f"Velocity (ux) at top center (Node {top_center_node_id}): {solution_vec[ux_top_dof]:.4f}")
    assert np.isclose(solution_vec[ux_top_dof], 1.0), "Lid velocity BC not applied correctly."

    bottom_center_node_id = -1
    for node in mesh_q2.nodes_list:
        if np.isclose(node.x, 0.5) and np.isclose(node.y, 0.0):
            bottom_center_node_id = node.id
            break
            
    ux_bottom_dof = dof_handler.dof_map['ux'][bottom_center_node_id]
    print(f"Velocity (ux) at bottom center (Node {bottom_center_node_id}): {solution_vec[ux_bottom_dof]:.4f}")
    assert np.isclose(solution_vec[ux_bottom_dof], 0.0), "No-slip BC not applied correctly."
    
    # Verify pressure pin
    p_pin_dof = dof_handler.dof_map['p'][0]
    print(f"Pressure at pinned node (Node 0): {solution_vec[p_pin_dof]:.4f}")
    assert np.isclose(solution_vec[p_pin_dof], 0.0), "Pressure pinning failed."

    print("\nStokes Q2-Q1 test checks passed successfully!")
