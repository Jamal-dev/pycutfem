import numpy as np, scipy.sparse.linalg as spla, sympy as sp
from pycutfem.utils.meshgen import structured_quad
from pycutfem.core import Mesh
from pycutfem.assembly.dg_global import assemble_dg

x, y = sp.symbols("x y")
u_sym = x**2 * y + sp.sin(y)
f_sym = -sp.diff(u_sym, x, 2) - sp.diff(u_sym, y, 2)
ue = sp.lambdify((x, y), u_sym, "numpy")
fe = sp.lambdify((x, y), f_sym, "numpy")


def test_sipg_q1():
    nodes, elems = structured_quad(1, 1, nx=8, ny=8, poly_order=1)
    mesh = Mesh(nodes, elems, element_type="quad",poly_order=1)
    K, F = assemble_dg(mesh, poly_order=1, penalty=100.0,dirichlet=lambda x, y: ue(x, y))
    # Source term (volume only) — quick loop
    from pycutfem.integration import volume
    from pycutfem.fem.reference import get_reference
    from pycutfem.fem import transform
    ref = get_reference("quad", 1)
    pts, wts = volume("quad", 3)
    n_loc = 4
    for eid, elem in enumerate(mesh.elements):
        dofs = np.arange(n_loc) + eid * n_loc
        Fe = np.zeros(n_loc)
        for (xi, eta), w in zip(pts, wts):
            N = ref.shape(xi, eta)
            J = transform.jacobian(mesh, eid, (xi, eta))
            detJ = abs(np.linalg.det(J))
            xphys = transform.x_mapping(mesh, eid, (xi, eta))
            Fe += w * detJ * N * fe(*xphys)
        F[dofs] += Fe
    uh = spla.spsolve(K, F)
    # Compute RMS nodal error (average of element nodal values)
    nodal_u = np.zeros(len(nodes)); counts = np.zeros(len(nodes))
    for eid, elem in enumerate(mesh.elements):
        dofs = np.arange(n_loc) + eid * n_loc
        nodal_u[elem] += uh[dofs]
        counts[elem] += 1
    nodal_u /= counts
    err = np.sqrt(np.mean((nodal_u - ue(nodes[:, 0], nodes[:, 1]))**2))
    assert err < 0.05  # DG is lower order, tolerance looser
