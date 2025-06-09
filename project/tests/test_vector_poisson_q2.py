import numpy as np, scipy.sparse.linalg as spla, sympy as sp
import scipy
from pycutfem.utils.meshgen import structured_quad
from pycutfem.core import Mesh
from pycutfem.assembly import stiffness_matrix, assemble
from pycutfem.assembly.load_vector import cg_element_load
from pycutfem.assembly.boundary_conditions import apply_dirichlet

x, y = sp.symbols("x y")
ux_sym = x**2 * y + sp.sin(y)
uy_sym = x * y**2 + sp.cos(x)
ux_rhs = -sp.diff(ux_sym, x, 2) - sp.diff(ux_sym, y, 2)
uy_rhs = -sp.diff(uy_sym, x, 2) - sp.diff(uy_sym, y, 2)
uex = sp.lambdify((x, y), ux_sym, "numpy")
uey = sp.lambdify((x, y), uy_sym, "numpy")
fx  = sp.lambdify((x, y), ux_rhs, "numpy")
fy  = sp.lambdify((x, y), uy_rhs, "numpy")

def test_vector_poisson_q2():
    nodes, elem_connectivity, edge_connectvity, elem_connectivity_corner_nodes = structured_quad(3, 2, nx=8, ny=5, poly_order=2)
    mesh = Mesh(nodes, elem_connectivity, edge_connectvity,elem_connectivity_corner_nodes, element_type="quad",poly_order=2)
    # Assemble block‑diagonal (2× scalar problems)
    K = assemble(mesh, lambda eid: stiffness_matrix(mesh, eid, quad_order=4))
    Kblock = scipy.sparse.block_diag((K, K))
    F = np.zeros(2 * len(nodes))
    for eid, elem in enumerate(mesh.elements_list):
        Fe_x = cg_element_load(mesh, eid, fx, poly_order=mesh.poly_order, quad_order=4)
        Fe_y = cg_element_load(mesh, eid, fy, poly_order=mesh.poly_order, quad_order=4)
        for a, A in enumerate(elem.nodes):
            F[A] += Fe_x[a]
            F[A + len(nodes)] += Fe_y[a]

    # Dirichlet on both components
    dbc = {}
    for dof, (xp, yp) in enumerate(mesh.nodes_x_y_pos):
        if np.isclose(xp, 0) | np.isclose(xp, 3) | np.isclose(yp, 0) | np.isclose(yp, 2):
            dbc[dof]                 = uex(xp, yp)
            dbc[dof + len(nodes)] = uey(xp, yp)

    Kbc, Fbc = apply_dirichlet(Kblock, F, dbc)
    uh = spla.spsolve(Kbc, Fbc)
    ux_h = uh[:len(nodes)]
    uy_h = uh[len(nodes):]
    err = np.sqrt(np.mean((ux_h - uex(mesh.nodes_x_y_pos[:, 0], mesh.nodes_x_y_pos[:, 1]))**2 +
                         (uy_h - uey(mesh.nodes_x_y_pos[:, 0], mesh.nodes_x_y_pos[:, 1]))**2))
    assert err < 1e-2
