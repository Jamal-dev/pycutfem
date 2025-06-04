import numpy as np, scipy.sparse.linalg as spla, sympy as sp
import scipy
from pycutfem.utils.meshgen import structured_quad
from pycutfem.core import Mesh
from pycutfem.assembly import stiffness_matrix, assemble
from pycutfem.assembly.load_vector import element_load
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
    nodes, elems = structured_quad(3, 2, nx=8, ny=5, element_order=2)
    mesh = Mesh(nodes, elems, element_type="quad",element_order=2)
    # Assemble block‑diagonal (2× scalar problems)
    K = assemble(mesh, lambda eid: stiffness_matrix(mesh, eid, order=4))
    Kblock = scipy.sparse.block_diag((K, K))
    F = np.zeros(2 * len(nodes))
    for eid, elem in enumerate(mesh.elements):
        Fe_x = element_load(mesh, eid, fx, order=4)
        Fe_y = element_load(mesh, eid, fy, order=4)
        for a, A in enumerate(elem):
            F[A] += Fe_x[a]
            F[A + len(nodes)] += Fe_y[a]

    # Dirichlet on both components
    dbc = {}
    for dof, (xp, yp) in enumerate(nodes):
        if np.isclose(xp, 0) | np.isclose(xp, 3) | np.isclose(yp, 0) | np.isclose(yp, 2):
            dbc[dof]                 = uex(xp, yp)
            dbc[dof + len(nodes)] = uey(xp, yp)

    Kbc, Fbc = apply_dirichlet(Kblock, F, dbc)
    uh = spla.spsolve(Kbc, Fbc)
    ux_h = uh[:len(nodes)]
    uy_h = uh[len(nodes):]
    err = np.sqrt(np.mean((ux_h - uex(nodes[:, 0], nodes[:, 1]))**2 +
                         (uy_h - uey(nodes[:, 0], nodes[:, 1]))**2))
    assert err < 1e-2
