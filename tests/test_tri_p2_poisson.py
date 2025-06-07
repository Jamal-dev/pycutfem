import numpy as np, scipy.sparse.linalg as spla, sympy as sp
from pycutfem.utils.meshgen import structured_triangles
from pycutfem.core import Mesh
from pycutfem.assembly import stiffness_matrix, assemble
from pycutfem.assembly.load_vector import cg_element_load
from pycutfem.assembly.boundary_conditions import apply_dirichlet

x, y = sp.symbols("x y")
# u_sym = x**2 * y + sp.sin(y)
u_sym = x**2  + y**2
f_sym = -sp.diff(u_sym, x, 2) - sp.diff(u_sym, y, 2)
u_exact = sp.lambdify((x, y), u_sym, "numpy")
f_exact = sp.lambdify((x, y), f_sym, "numpy")

def test_poisson_p2():
    poly_order = 2
    quad_order = poly_order+2
    nodes, elems, edge_connectvity, elem_connectivity_corner_nodes = structured_triangles(3, 2,nx_quads=12, ny_quads=9, poly_order=poly_order)
    mesh = Mesh(nodes, elems, edge_connectvity, elem_connectivity_corner_nodes, element_type="tri", poly_order=poly_order)
    K = assemble(mesh, lambda eid: stiffness_matrix(mesh, eid, quad_order=quad_order))
    F = np.zeros(len(nodes))
    for eid, elem in enumerate(mesh.elements_list):
        Fe = cg_element_load(mesh, eid, f_exact, poly_order=mesh.poly_order, quad_order=quad_order)
        for a, A in enumerate(elem.nodes):
            F[A] += Fe[a]
    dbc = {dof: u_exact(xp, yp) for dof, (xp, yp) in enumerate(mesh.nodes)
           if np.isclose(xp, 0) | np.isclose(xp, 3) | np.isclose(yp, 0) | np.isclose(yp, 2)}
    K, F = apply_dirichlet(K, F, dbc)
    uh = spla.spsolve(K, F)
    err = np.sqrt(np.mean((uh - u_exact(mesh.nodes[:, 0], mesh.nodes[:, 1]))**2))
    assert err < 1e-2

def test_poisson_p1():
    poly_order = 1
    quad_order = 5
    nodes, elems, edge_connectvity, elem_connectivity_corner_nodes = structured_triangles(3, 2,nx_quads=12, ny_quads=9, poly_order=poly_order)
    mesh = Mesh(nodes, elems, edge_connectvity, elem_connectivity_corner_nodes, element_type="tri", poly_order=poly_order)
    K = assemble(mesh, lambda eid: stiffness_matrix(mesh, eid, quad_order=quad_order))
    F = np.zeros(len(nodes))
    for eid, elem in enumerate(mesh.elements_list):
        Fe = cg_element_load(mesh, eid, f_exact, poly_order=mesh.poly_order, quad_order=quad_order)
        for a, A in enumerate(elem.nodes):
            F[A] += Fe[a]
    dbc = {dof: u_exact(xp, yp) for dof, (xp, yp) in enumerate(mesh.nodes)
           if np.isclose(xp, 0) | np.isclose(xp, 3) | np.isclose(yp, 0) | np.isclose(yp, 2)}
    K, F = apply_dirichlet(K, F, dbc)
    uh = spla.spsolve(K, F)
    err = np.sqrt(np.mean((uh - u_exact(mesh.nodes[:, 0], mesh.nodes[:, 1]))**2))
    assert err < 1e-2


# def test_poisson_p3():
#     poly_order = 3
#     quad_order = poly_order + 2
#     nodes, elems, edge_connectvity, elem_connectivity_corner_nodes = structured_triangles(3, 2,nx_quads=12, ny_quads=9, poly_order=poly_order)
#     mesh = Mesh(nodes, elems, edge_connectvity, elem_connectivity_corner_nodes, element_type="tri", poly_order=poly_order)
#     K = assemble(mesh, lambda eid: stiffness_matrix(mesh, eid, quad_order=quad_order))
#     F = np.zeros(len(nodes))
#     for eid, elem in enumerate(mesh.elements):
#         Fe = cg_element_load(mesh, eid, f_exact, poly_order=mesh.poly_order, quad_order=quad_order)
#         for a, A in enumerate(elem):
#             F[A] += Fe[a]
#     dbc = {dof: u_exact(xp, yp) for dof, (xp, yp) in enumerate(nodes)
#            if np.isclose(xp, 0) | np.isclose(xp, 3) | np.isclose(yp, 0) | np.isclose(yp, 2)}
#     K, F = apply_dirichlet(K, F, dbc)
#     uh = spla.spsolve(K, F)
#     err = np.sqrt(np.mean((uh - u_exact(nodes[:, 0], nodes[:, 1]))**2))
#     assert err < 1e-2