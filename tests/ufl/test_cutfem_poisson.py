from matplotlib.pylab import norm
from pycutfem.utils import boundary
from pyparsing import C
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
from pycutfem.ufl.expressions import (TrialFunction, TestFunction, grad, inner, jump,dot,
                             ElementWiseConstant, 
                             Jump,
                             Constant, FacetNormal, CellDiameter, Derivative, Pos, Neg)
from pycutfem.ufl.measures import dx, ds, dInterface, dGhost
from pycutfem.ufl.forms import BoundaryCondition, assemble_form, Equation

# --- Level Set and Cut Ratio imports ---
from pycutfem.core.levelset import CircleLevelSet
from pycutfem.utils.geometry import hansbo_cut_ratio
from pycutfem.io.visualization import plot_mesh_2
from pycutfem.fem.mixedelement import MixedElement

def _dn(expr,n):
    """Normal derivative  n·∇expr  on an (interior) edge."""
    return n[0] * Derivative(expr, 1, 0) + n[1] * Derivative(expr, 0, 1)
    # return dot(grad(expr), n)

def grad_inner(u, v, n):
    """⟨∂ₙu, ∂ₙv⟩  (scalar or 2‑D vector)."""
    if getattr(u, "num_components", 1) == 1:      # scalar
        return _dn(u, n) * _dn(v, n)

    if u.num_components == v.num_components == 2: # vector
        return _dn(u[0], n) * _dn(v[0], n) + _dn(u[1], n) * _dn(v[1], n)


def test_cutfem_poisson_interface():
    """
    Tests the solver for a Poisson interface problem using CutFEM.
    -α∇²u = f
    This version uses the final, robust, BitSet-driven framework.
    """
    # 1. Setup Mesh and Level Set
    poly_order = 1
    L,H = 2.0, 2.0
    ghost_parameter = 0.5
    gamma_G = Constant(ghost_parameter)
    nodes, elems, _, corners = structured_quad(L,H, nx=10, ny=10, poly_order=poly_order)
    mesh = Mesh(nodes, element_connectivity=elems, elements_corner_nodes=corners, poly_order=poly_order, element_type='quad')
    c_x, c_y = L/2, H/2
    radius = 0.5
    level_set = CircleLevelSet(center=(c_x, c_y), radius=radius)

    # 2. Apply general boundary tags FIRST
    boundary_tags = {
        'left': lambda x, y: np.isclose(x, 0.0),
        'right': lambda x, y: np.isclose(x, L),
        'bottom': lambda x, y: np.isclose(y, 0.0),
        'top': lambda x, y: np.isclose(y, H),
    }
    mesh.tag_boundary_edges(boundary_tags)
    # Classify mesh elements and edges against the level set
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)

    # 2. Create BitSets for the integration domains
    pos_elements = get_domain_bitset(mesh, 'element', 'outside')
    neg_elements = get_domain_bitset(mesh, 'element', 'inside')
    cut_elements = get_domain_bitset(mesh, 'element', 'cut')
    has_pos_elements = pos_elements | cut_elements
    has_neg_elements = neg_elements | cut_elements
    cut_domain = mesh.element_bitset("cut")
    # Ghost penalty is applied on faces between cut cells and interior cells
    ghost_domain = mesh.edge_bitset("ghost_neg") | mesh.edge_bitset("ghost_both")
    print(f"Negative elements: {neg_elements}, Positive elements: {pos_elements}, Cut elements: {cut_elements}")
    print(f"Has positive elements: {has_pos_elements}, Has negative elements: {has_neg_elements}")
    for tag, bitset in mesh._edge_bitsets.items():
        print(f"Boundary tag '{tag}': {bitset.cardinality()}")
    # plot_mesh_2(mesh,level_set=level_set)
    
    # Interface edges for the `ds` integral
    if_edges = get_domain_bitset(mesh, 'edge', 'interface')

    # 3. Define the DofHandler for the two-field problem
    # The field names are now just labels; their meaning is defined by their use in the weak form.
    me = MixedElement(mesh, field_specs={'u_outside' : poly_order, 'u_inside': poly_order})
    # fe_map = {'u_outside': mesh, 'u_inside': mesh}
    dof_handler = DofHandler(me, method='cg')

    dof_handler.tag_dofs_from_element_bitset("inactive_inside", "u_outside", "inside", strict=True)
    dof_handler.tag_dofs_from_element_bitset("inactive_outside", "u_inside", "outside", strict=True)


    # 4. Define Trial and Test Functions
    u_pos, v_pos = TrialFunction('u_outside',dof_handler=dof_handler), TestFunction('u_outside',dof_handler=dof_handler)
    u_neg, v_neg = TrialFunction('u_inside',dof_handler=dof_handler), TestFunction('u_inside',dof_handler=dof_handler)

    # 5. Define Coefficients and Expressions
    alpha_vals = np.zeros(len(mesh.elements_list))
    alpha_vals[pos_elements.to_indices()] = 1.0
    alpha_vals[neg_elements.to_indices()] = 20.0
    alpha_vals[cut_elements.to_indices()] = 1.0 # Convention: use alpha_neg on cut elements
    alpha = ElementWiseConstant(alpha_vals)
    
    h = CellDiameter()
    stab = Constant(20 * (20.0 + 1.0)) / h

    # The jump operator is now the explicit source of truth for +/- sides.
    jump_u = jump(u_neg, u_pos)
    jump_v = jump(v_neg, v_pos)
    
    normal = FacetNormal()
    alpha_pos = Pos(alpha)   # α(+)
    alpha_neg = Neg(alpha)   # α(-)
    # We define the average flux using the functions themselves
    # n is oriented from (−) to (+); so ∂ₙu(−) uses −n
    avg_flux_u = -0.5 * ( alpha_pos * dot(grad(u_pos), normal)
                        - alpha_neg * dot(grad(u_neg), normal) )
    avg_flux_v = -0.5 * ( alpha_pos * dot(grad(v_pos), normal)
                        - alpha_neg * dot(grad(v_neg), normal) )
    # 6. Define the Weak Form
    # Volume terms are integrated over their respective domains (including cut elements)
    dx_pos = dx(defined_on=has_pos_elements, level_set=level_set,  metadata={'side': '+'})
    dx_neg = dx(defined_on=has_neg_elements, level_set=level_set,  metadata={'side': '-'})
    dGamma = dInterface(defined_on=cut_domain, level_set=level_set, metadata={"q": poly_order + 2})
    dGhost_stab = dGhost(defined_on=ghost_domain, level_set=level_set, metadata={"q": poly_order + 2})
    a_stabilization = (0.5 * gamma_G * h *
                           grad_inner(Jump(u_pos,u_neg),Jump(v_pos,v_neg),normal)* dGhost_stab)
    a =  inner(alpha * grad(u_pos), grad(v_pos)) * dx_pos
    a += inner(alpha * grad(u_neg), grad(v_neg)) * dx_neg
    a += a_stabilization

    # Interface terms use the ds measure, which now requires the level_set for orientation
    a += ( dot(avg_flux_u, jump_v) + dot(avg_flux_v, jump_u) + stab * jump_u * jump_v ) * dGamma
    # a += ( avg_flux_u * jump_v + avg_flux_v * jump_u + stab * jump_u * jump_v ) * dGamma


    # Right-hand side
    f =  Constant(1.0) * v_pos * dx_pos
    f += Constant(1.0) * v_neg * dx_neg

    equation = (Equation(a,f))

    # 7. Define Boundary Conditions
    bcs = [
        BoundaryCondition('u_outside', 'dirichlet', 'inactive_inside', lambda x, y: 0.0),
        BoundaryCondition('u_inside', 'dirichlet' , 'inactive_outside', lambda x, y: 0.0),
        BoundaryCondition('u_outside', 'dirichlet', 'left', lambda x, y: 0.0),
        BoundaryCondition('u_outside', 'dirichlet', 'right', lambda x, y: 0.0),
        BoundaryCondition('u_outside', 'dirichlet', 'bottom', lambda x, y: 0.0),
        BoundaryCondition('u_outside', 'dirichlet', 'top', lambda x, y: 0.0),
        BoundaryCondition('u_inside', 'dirichlet', 'left', lambda x, y: 0.0),
        BoundaryCondition('u_inside', 'dirichlet', 'right', lambda x, y: 0.0),
        BoundaryCondition('u_inside', 'dirichlet', 'bottom', lambda x, y: 0.0),
        BoundaryCondition('u_inside', 'dirichlet', 'top', lambda x, y: 0.0),    
    ]
    
    # 8. Assemble and Solve
    system = equation
    K, F = assemble_form(system, dof_handler=dof_handler, bcs=bcs, quad_order=poly_order + 2)

    free = np.where(K.getnnz(axis=1) == 0)[0]
    print("Zero-row DOFs:", free)


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