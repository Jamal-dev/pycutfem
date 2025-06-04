import numpy as np, scipy.sparse.linalg as spla, sympy as sp
from pycutfem.utils.meshgen import structured_quad
from pycutfem.core import Mesh
from pycutfem.assembly import stiffness_matrix, assemble
from pycutfem.assembly.load_vector import element_load
from pycutfem.assembly.boundary_conditions import apply_dirichlet

# Manufactured solution (scalar) using SymPy
x, y = sp.symbols("x y")
u_sym = x**2 * y + sp.sin(y)
f_sym = -sp.diff(u_sym, x, 2) - sp.diff(u_sym, y, 2)
ue = sp.lambdify((x, y), u_sym, "numpy")
fe = sp.lambdify((x, y), f_sym, "numpy")

def solve_q2():
    nodes, elems = structured_quad(3, 2, nx=10, ny=6, element_order=2)
    mesh = Mesh(nodes, elems, element_type="quad",element_order=2)
    K = assemble(mesh, lambda eid: stiffness_matrix(mesh, eid, order=4))
    F = np.zeros(len(nodes))
    for eid, elem in enumerate(mesh.elements):
        Fe = element_load(mesh, eid, fe, order=4)
        for a, A in enumerate(elem):
            F[A] += Fe[a]

    dbc = {dof: ue(xp, yp) for dof, (xp, yp) in enumerate(nodes)
           if np.isclose(xp, 0) | np.isclose(xp, 3) | np.isclose(yp, 0) | np.isclose(yp, 2)}
    K, F = apply_dirichlet(K, F, dbc)
    uh = spla.spsolve(K, F)
    err = np.sqrt(np.mean((uh - ue(nodes[:, 0], nodes[:, 1]))**2))
    return err

def test_poisson_q2():
    assert solve_q2() < 5e-3
