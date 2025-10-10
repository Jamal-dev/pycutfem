import numpy as np, scipy.sparse.linalg as spla, sympy as sp
from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.assembly import stiffness_matrix, assemble
from pycutfem.assembly.load_vector import cg_element_load
from pycutfem.assembly.boundary_conditions import apply_dirichlet

# Manufactured solution (scalar) using SymPy
x, y = sp.symbols("x y")
u_sym = x**2 * y + sp.sin(y)
f_sym = -sp.diff(u_sym, x, 2) - sp.diff(u_sym, y, 2)
ue = sp.lambdify((x, y), u_sym, "numpy")
fe = sp.lambdify((x, y), f_sym, "numpy")

def solve(poly_order=2):
    quad_order = poly_order + 3
    nodes, elems, edge_connectvity, elem_connectivity_corner_nodes = structured_quad(3, 2, nx=10, ny=6, poly_order=poly_order)
    mesh = Mesh(nodes, elems, edge_connectvity, elem_connectivity_corner_nodes, element_type="quad",poly_order=poly_order)
    K = assemble(mesh, lambda eid: stiffness_matrix(mesh, eid, quad_order=quad_order))
    F = np.zeros(len(nodes))
    for eid, elem in enumerate(mesh.elements_list):
        Fe = cg_element_load(mesh, eid, fe,poly_order=mesh.poly_order, quad_order=quad_order)
        for a, A in enumerate(elem.nodes):
            F[A] += Fe[a]

    dbc = {dof: ue(xp, yp) for dof, (xp, yp) in enumerate(mesh.nodes_x_y_pos)
           if np.isclose(xp, 0) | np.isclose(xp, 3) | np.isclose(yp, 0) | np.isclose(yp, 2)}
    K, F = apply_dirichlet(K, F, dbc)
    uh = spla.spsolve(K, F)
    err = np.sqrt(np.mean((uh - ue(mesh.nodes_x_y_pos[:, 0], mesh.nodes_x_y_pos[:, 1]))**2))
    return err

def test_poisson_q2():
    assert solve(2) < 5e-3

# def test_poisson_q3():
#     assert solve(3) < 5e-3
