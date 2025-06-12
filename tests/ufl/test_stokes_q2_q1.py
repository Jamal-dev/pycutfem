import pytest
import numpy as np
import scipy.sparse.linalg as spla

# --- Core imports from the refactored library ---
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad

# --- UFL-like imports ---
from ufl.functionspace import FunctionSpace
from ufl.expressions import (
    TrialFunction, TestFunction, VectorTrialFunction, VectorTestFunction,
    grad, inner, dot, div, Constant
)
from ufl.measures import dx
from ufl.forms import BoundaryCondition, assemble_form

def test_stokes_lid_driven_cavity():
    """
    Tests the solver for the Stokes system using Q2-Q1 Taylor-Hood elements.
    This solves the lid-driven cavity problem using a full vector form.
    """
    # 1. Setup Meshes and DofHandler for Q2-Q1 elements
    nodes_q2, elems_q2, _, corners_q2 = structured_quad(1, 1, nx=4, ny=4, poly_order=2)
    mesh_q2 = Mesh(nodes=nodes_q2, element_connectivity=elems_q2,
                   elements_corner_nodes=corners_q2, element_type="quad", poly_order=2)

    nodes_q1, elems_q1, _, corners_q1 = structured_quad(1, 1, nx=4, ny=4, poly_order=1)
    mesh_q1 = Mesh(nodes=nodes_q1, element_connectivity=elems_q1,
                   elements_corner_nodes=corners_q1, element_type="quad", poly_order=1)

    fe_map = {
        'ux': mesh_q2, 'uy': mesh_q2, 'p': mesh_q1
    }
    dof_handler = DofHandler(fe_map, method='cg')

    # 2. Define UFL symbols using vector notation
    velocity_space = FunctionSpace("velocity", ['ux', 'uy'], dim=1)
    pressure_space = FunctionSpace("pressure", ['p'], dim=0)

    u = VectorTrialFunction(velocity_space)
    v = VectorTestFunction(velocity_space)
    p = TrialFunction(pressure_space.field_names[0])
    q = TestFunction(pressure_space.field_names[0])


    # 3. Define the Weak Form in vector notation
    # ∫ (∇u : ∇v - p(∇⋅v) + q(∇⋅u)) dx = ∫ f⋅v dx
    a_form = (inner(grad(u), grad(v)) - p * div(v) + q * div(u)) * dx()
    f = Constant((0.0, 0.0))
    L_form = dot(f, v) * dx()
    
    equation = a_form == L_form

    # 4. Define Boundary Conditions for the Lid-Driven Cavity
    bc_tags = {
        'bottom_wall': lambda x, y: np.isclose(y, 0),
        'left_wall':   lambda x, y: np.isclose(x, 0),
        'right_wall':  lambda x, y: np.isclose(x, 1),
        'top_lid':     lambda x, y: np.isclose(y, 1)
    }
    mesh_q2.tag_boundary_edges(bc_tags)
    mesh_q1.nodes_list[0].tag = 'pressure_pin_point'

    bcs = [
        BoundaryCondition('ux', 'dirichlet', 'left_wall',   lambda x, y: 0.0),
        BoundaryCondition('uy', 'dirichlet', 'left_wall',   lambda x, y: 0.0),
        BoundaryCondition('ux', 'dirichlet', 'right_wall',  lambda x, y: 0.0),
        BoundaryCondition('uy', 'dirichlet', 'right_wall',  lambda x, y: 0.0),
        BoundaryCondition('ux', 'dirichlet', 'bottom_wall', lambda x, y: 0.0),
        BoundaryCondition('uy', 'dirichlet', 'bottom_wall', lambda x, y: 0.0),
        BoundaryCondition('ux', 'dirichlet', 'top_lid',     lambda x, y: 1.0),
        BoundaryCondition('uy', 'dirichlet', 'top_lid',     lambda x, y: 0.0),
        BoundaryCondition('p', 'dirichlet', 'pressure_pin_point', lambda x, y: 0.0)
    ]

    # 5. Assemble and Solve the System
    K, F = assemble_form(equation, dof_handler=dof_handler, bcs=bcs, quad_order=5)

    matrix_rank = np.linalg.matrix_rank(K.toarray())
    print(f"\n[Lid-Driven] Matrix Info: Shape={K.shape}, Rank={matrix_rank}")
    assert matrix_rank == K.shape[0], "Matrix is rank-deficient!"

    solution_vec = spla.spsolve(K, F)

    # 6. Verify Solution
    ux_top_dof = dof_handler.dof_map['ux'][mesh_q2.nodes_list[-5].id] # A node on top edge
    assert np.isclose(solution_vec[ux_top_dof], 1.0), "Lid velocity BC not applied correctly."
    p_pin_dof = dof_handler.dof_map['p'][0]
    assert np.isclose(solution_vec[p_pin_dof], 0.0), "Pressure pinning failed."
    print("\nStokes Q2-Q1 Lid-Driven Cavity test passed successfully!")


def test_stokes_couette_flow_vector_form():
    """
    Tests the solver for Stokes flow using a full vector formulation.
    This solves Couette flow and compares to the known analytical solution.
    """
    # 1. Setup: Use same Q2-Q1 elements
    print("\n" + "="*70)
    print("Testing Stokes Couette Flow with Vector Formulation")
    print("="*70)
    
    nx, ny = 4, 8 # Use a finer mesh in y for better profile capture
    nodes_q2, elems_q2, _, c_q2 = structured_quad(1, 1, nx=nx, ny=ny, poly_order=2)
    mesh_q2 = Mesh(nodes=nodes_q2, element_connectivity=elems_q2, elements_corner_nodes=c_q2, element_type="quad", poly_order=2)

    nodes_q1, elems_q1, _, c_q1 = structured_quad(1, 1, nx=nx, ny=ny, poly_order=1)
    mesh_q1 = Mesh(nodes=nodes_q1, element_connectivity=elems_q1, elements_corner_nodes=c_q1, element_type="quad", poly_order=1)

    fe_map = {'ux': mesh_q2, 'uy': mesh_q2, 'p': mesh_q1}
    dof_handler = DofHandler(fe_map, method='cg')

    # 2. Define UFL symbols using vector notation
    velocity_space = FunctionSpace("velocity", ['ux', 'uy'], dim=1)
    pressure_space = FunctionSpace("pressure", ['p'], dim=0)

    u = VectorTrialFunction(velocity_space)
    v = VectorTestFunction(velocity_space)
    p = TrialFunction(pressure_space.field_names[0])
    q = TestFunction(pressure_space.field_names[0])

    # 3. Define the Weak Form in vector notation
    # ∫ (∇u : ∇v - p(∇⋅v) + q(∇⋅u)) dx = ∫ f⋅v dx
    a = (inner(grad(u), grad(v)) - p * div(v) + q * div(u)) * dx()
    f = Constant((0.0, 0.0))
    L = dot(f, v) * dx()
    equation = (a == L)

    # 4. Define Boundary Conditions for Couette Flow
    bc_tags = {'bottom': lambda x,y: np.isclose(y,0), 'top': lambda x,y: np.isclose(y,1)}
    mesh_q2.tag_boundary_edges(bc_tags)
    mesh_q1.nodes_list[0].tag = 'pressure_pin_point' # Pin pressure at origin

    bcs = [
        # Stationary bottom wall (no-slip)
        BoundaryCondition('ux', 'dirichlet', 'bottom', lambda x, y: 0.0),
        BoundaryCondition('uy', 'dirichlet', 'bottom', lambda x, y: 0.0),
        # Moving top wall (U_top = 1.0)
        BoundaryCondition('ux', 'dirichlet', 'top',    lambda x, y: 1.0),
        BoundaryCondition('uy', 'dirichlet', 'top',    lambda x, y: 0.0),
        # Pin pressure to ensure a unique solution
        BoundaryCondition('p', 'dirichlet', 'pressure_pin_point', lambda x, y: 0.0)
    ]

    # 5. Assemble and Solve
    K, F = assemble_form(equation, dof_handler, bcs=bcs, quad_order=5)
    
    matrix_rank = np.linalg.matrix_rank(K.toarray())
    print(f"\n[Couette] Matrix Info: Shape={K.shape}, Rank={matrix_rank}")
    assert matrix_rank == K.shape[0], "Matrix is rank-deficient!"

    solution_vec = spla.spsolve(K, F)

    # 6. Verification against analytical solution: ux = y, uy = 0
    print("\nVerifying solution against analytical profile ux(y) = y...")
    TOL = 1e-9
    for node in mesh_q2.nodes_list:
        # Check ux profile
        ux_dof = dof_handler.dof_map['ux'][node.id]
        computed_ux = solution_vec[ux_dof]
        exact_ux = node.y # Since U_top=1 and H=1
        
        # We only check the profile on nodes not directly on the BC boundaries
        # to ensure the solver is producing the correct internal values.
        if not (np.isclose(node.y, 0.0) or np.isclose(node.y, 1.0)):
             assert np.isclose(computed_ux, exact_ux, atol=TOL), \
                f"Incorrect ux at y={node.y:.2f}. Expected {exact_ux:.4f}, got {computed_ux:.4f}"

        # Check uy is zero everywhere
        uy_dof = dof_handler.dof_map['uy'][node.id]
        computed_uy = solution_vec[uy_dof]
        print(f"{node}")
        assert np.isclose(computed_uy, 0.0, atol=TOL), \
            f"Non-zero uy at ({node.x:.2f}, {node.y:.2f}). Got {computed_uy:.4f}"
            
    print("Couette flow velocity profile matches analytical solution.")
    print("Vector formulation test passed successfully!")

if __name__ == "__main__":
    # To run the tests, use pytest from your terminal:
    # > pytest test_stokes_flow.py
    test_stokes_lid_driven_cavity()
    test_stokes_couette_flow_vector_form()
