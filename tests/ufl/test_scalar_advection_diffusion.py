import pytest
import numpy as np
import scipy.sparse.linalg as spla
import sympy as sp

# --- Imports from your project structure ---
from pycutfem.core import Mesh, Node
from pycutfem.utils.meshgen import structured_quad
from ufl.functionspace import FunctionSpace
from ufl.expressions import TrialFunction, TestFunction, grad, inner, dot, Constant
from ufl.measures import dx
from ufl.forms import assemble_form, BoundaryCondition
from ufl.analytic import Analytic, x, y

def test_advection_diffusion_solve():
    """
    Tests the symbolic solver for a steady-state advection-diffusion problem.
    -∇⋅(ε∇u) + β⋅∇u = f
    """
    # 1. Define Analytical Solution and Parameters
    beta_vec = np.array([1.0, 1.0])
    epsilon = 0.01
    
    u_exact_sym = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    f_sym = -epsilon*sp.diff(u_exact_sym, x, 2) - epsilon*sp.diff(u_exact_sym, y, 2) + \
            beta_vec[0]*sp.diff(u_exact_sym, x) + beta_vec[1]*sp.diff(u_exact_sym, y)

    u_exact = Analytic(u_exact_sym)
    f = Analytic(f_sym)

    # 2. Setup Mesh and Function Space
    nodes_coords, elems, _, corners = structured_quad(1, 1, nx=8, ny=8, poly_order=2)
    nodes_list = [Node(id=i, x=c[0], y=c[1]) for i,c in enumerate(nodes_coords)]
    mesh = Mesh(nodes=nodes_list, element_connectivity=elems, 
                elements_corner_nodes=corners, element_type="quad", poly_order=2)

    V = FunctionSpace(mesh, p=2, dim=1) # Scalar problem

    # 3. Define the Weak Form
    u, v = TrialFunction(V), TestFunction(V)
    beta = Constant(beta_vec)

    # Weak form: ∫(ε∇v⋅∇u + (β⋅∇u)v)dx = ∫fv dx
    a = (epsilon * inner(grad(u), grad(v)) + dot(beta, grad(u)) * v) * dx()
    L = f * v * dx()
    
    equation = (a == L)

    # 4. Define Boundary Conditions
    # We strongly impose Dirichlet BCs on the whole boundary for this test
    mesh.tag_boundary_edges({'boundary': lambda x,y: True})
    bcs = [BoundaryCondition(V, 'dirichlet', 'boundary', u_exact)]

    # 5. Assemble and Solve
    K, F = assemble_form(equation, function_space=V,mesh=mesh, bcs=bcs, quad_order=4)
    uh = spla.spsolve(K, F)

    # 6. Verify Solution
    # We approximate the L2 error by checking the error at the nodes
    nodal_uh = np.zeros(len(mesh.nodes_list))
    counts = np.zeros(len(mesh.nodes_list))
    n_loc = V.num_local_dofs()

    for eid, elem in enumerate(mesh.elements_list):
        dofs = np.arange(n_loc) + eid * n_loc
        nodal_uh[list(elem.nodes)] += uh[dofs]
        counts[list(elem.nodes)] += 1
    nodal_uh /= counts
    
    exact_vals_at_nodes = u_exact.eval(mesh.nodes_x_y_pos)
    l2_error = np.sqrt(np.mean((nodal_uh - exact_vals_at_nodes)**2))
    
    print(f"\nL2 Error for Advection-Diffusion problem: {l2_error:.4e}")
    assert l2_error < 0.05, "Solution error is too high."

