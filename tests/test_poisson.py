
import numpy as np
import scipy.sparse.linalg as spla
import sympy as sp

from pycutfem.utils.meshgen import structured_quad, delaunay_rectangle
from pycutfem.core import Mesh
from pycutfem.assembly import stiffness_matrix, assemble
from pycutfem.assembly.load_vector import cg_element_load
from pycutfem.assembly.boundary_conditions import apply_dirichlet

# SymPy exact solution and RHS
x, y = sp.symbols('x y')
u_sym = x**2 * y + sp.sin(y)
f_sym = -sp.diff(u_sym, x, 2) - sp.diff(u_sym, y, 2)
u_exact = sp.lambdify((x, y), u_sym, 'numpy')
f_rhs   = sp.lambdify((x, y), f_sym, 'numpy')

def solve(mesh):
    K = assemble(mesh, lambda eid: stiffness_matrix(mesh, eid,quad_order=5))
    F = np.zeros(len(mesh.nodes))
    for eid, elem in enumerate(mesh.elements):
        Fe = cg_element_load(mesh, eid, f_rhs, poly_order = mesh.poly_order)
        for a,A in enumerate(elem): F[A] += Fe[a]

    dbc={}
    for dof,(xp,yp) in enumerate(mesh.nodes):
        if np.isclose(xp,0)|np.isclose(xp,3)|np.isclose(yp,0)|np.isclose(yp,2):
            dbc[dof]=u_exact(xp,yp)
    K_bc, F_bc = apply_dirichlet(K, F, dbc)
    uh = spla.spsolve(K_bc, F_bc)
    return uh

def l2_error(mesh, uh):
    return np.sqrt(np.mean((uh - u_exact(mesh.nodes[:,0], mesh.nodes[:,1]))**2))

def test_poisson_quad():
    nodes, elems = structured_quad(3,2, nx=20, ny=15, poly_order=1)
    mesh = Mesh(nodes, elems, element_type='quad', poly_order=1)
    uh = solve(mesh)
    assert l2_error(mesh, uh) < 2e-2

def test_poisson_tri():
    nodes, elems = delaunay_rectangle(3,2, nx=20, ny=15)
    mesh = Mesh(nodes, elems, element_type='tri', poly_order=1)
    uh = solve(mesh)
    assert l2_error(mesh, uh) < 2e-2
