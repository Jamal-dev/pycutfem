import pytest
import numpy as np
import scipy.sparse.linalg as spla

# --- Core imports ---
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.utils.bitset import BitSet
from pycutfem.utils.domain_manager import get_domain_bitset

# --- UFL-like imports ---
from ufl.expressions import (TrialFunction, TestFunction, grad, inner, jump,dot,
                             ElementWiseConstant, Constant)
from ufl.measures import dx, ds, dInterface
from ufl.forms import BoundaryCondition, assemble_form

# --- Level Set and Cut Ratio imports ---
from pycutfem.core.levelset import CircleLevelSet
from pycutfem.utils.geometry import hansbo_cut_ratio
from pycutfem.io.visualization import plot_mesh_2


def test_cutfem_poisson_interface():
    """
    Tests the solver for a Poisson interface problem using CutFEM.
    -α∇²u = f
    This version uses the final, robust, BitSet-driven framework.
    """
    # 1. Setup Mesh and Level Set
    poly_order = 1
    L,H = 2.0, 2.0
    nodes, elems, _, corners = structured_quad(L,H, nx=10, ny=10, poly_order=poly_order)
    mesh = Mesh(nodes, element_connectivity=elems, elements_corner_nodes=corners, poly_order=poly_order, element_type='quad')
    
    level_set = CircleLevelSet(center=(L/2,H/2), radius=0.5)

    # Classify mesh elements and edges against the level set
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)

    # 2. Create BitSets for the integration domains
    neg_elements = get_domain_bitset(mesh, 'element', 'outside')
    pos_elements = get_domain_bitset(mesh, 'element', 'inside')
    cut_elements = get_domain_bitset(mesh, 'element', 'cut')
    has_pos_elements = pos_elements | cut_elements
    has_neg_elements = neg_elements | cut_elements
    print(f"Negative elements: {neg_elements}, Positive elements: {pos_elements}, Cut elements: {cut_elements}")
    print(f"Has positive elements: {has_pos_elements}, Has negative elements: {has_neg_elements}")
    # plot_mesh_2(mesh,level_set=level_set)
    
    # Interface edges for the `ds` integral
    if_edges = get_domain_bitset(mesh, 'edge', 'interface')

    # 3. Define the DofHandler for the two-field problem
    # The field names are now just labels; their meaning is defined by their use in the weak form.
    fe_map = {'u_outside': mesh, 'u_inside': mesh}
    dof_handler = DofHandler(fe_map, method='cg')

    # 4. Define Trial and Test Functions
    u_neg, v_neg = TrialFunction('u_outside'), TestFunction('u_outside')
    u_pos, v_pos = TrialFunction('u_inside'), TestFunction('u_inside')

    # 5. Define Coefficients and Expressions
    alpha_vals = np.zeros(len(mesh.elements_list))
    alpha_vals[neg_elements.to_indices()] = 1.0
    alpha_vals[pos_elements.to_indices()] = 20.0
    alpha_vals[cut_elements.to_indices()] = 1.0 # Convention: use alpha_neg on cut elements
    alpha = ElementWiseConstant(alpha_vals)
    
    h = np.mean([np.sqrt(area) for area in mesh.areas()])
    stab = 20 * (20.0 + 1.0) / h

    # The jump operator is now the explicit source of truth for +/- sides.
    jump_u = jump(u_pos, u_neg)
    jump_v = jump(v_pos, v_neg)
    
    # We define the average flux using the functions themselves
    avg_flux_u = - (alpha * grad(u_neg) + alpha * grad(u_pos)) * 0.5
    avg_flux_v = - (alpha * grad(v_neg) + alpha * grad(v_pos)) * 0.5
    
    # 6. Define the Weak Form
    # Volume terms are integrated over their respective domains (including cut elements)
    a = inner(alpha * grad(u_neg), grad(v_neg)) * dx(defined_on=neg_elements | cut_elements)
    a += inner(alpha * grad(u_pos), grad(v_pos)) * dx(defined_on=pos_elements | cut_elements)
    
    # Interface terms use the ds measure, which now requires the level_set for orientation
    a += ( dot(avg_flux_u, jump_v) + dot(avg_flux_v, jump_u) + stab * jump_u * jump_v ) * dInterface( level_set=level_set)

    # Right-hand side
    f = Constant(1.0) * v_neg * dx(defined_on=neg_elements | cut_elements)
    f += Constant(1.0) * v_pos * dx(defined_on=pos_elements | cut_elements)
    
    equation = (a == f)

    # 7. Define Boundary Conditions
    mesh.tag_boundary_edges({'boundary': lambda x,y: True})
    bcs = [
        BoundaryCondition('u_outside', 'dirichlet', 'boundary', lambda x, y: 0.0),
        BoundaryCondition('u_inside', 'dirichlet', 'boundary', lambda x, y: 0.0),
    ]
    
    # 8. Assemble and Solve
    system = equation
    K, F = assemble_form(system, dof_handler=dof_handler, bcs=bcs, quad_order=poly_order + 2)

    # Check the rank to ensure the system is solvable
    matrix_rank = np.linalg.matrix_rank(K.toarray())
    print(f"\nMatrix Info: Shape={K.shape}, Rank={matrix_rank}")
    assert matrix_rank == K.shape[0], "Matrix is rank-deficient!"

    solution_vec = spla.spsolve(K, F)

    # 9. Verification
    assert np.linalg.norm(solution_vec) > 1e-8, "Solution is trivial (zero)."
    
    center_node_id = -1
    for i, node_data in enumerate(mesh.nodes_list):
        if np.isclose(node_data.x, L/2) and np.isclose(node_data.y, H/2):
            center_node_id = i
            break
            
    assert center_node_id != -1, "Could not find the center node of the mesh."

    u_pos_dof = dof_handler.dof_map['u_inside'][center_node_id]
    center_val = solution_vec[u_pos_dof]
    print(f"Solution at center (inside domain): {center_val:.4f}")
    assert center_val > 0, "Expected a positive solution at the center."