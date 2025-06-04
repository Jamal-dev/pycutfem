
"""Example: Poisson on structured quad mesh"""
import numpy as np, scipy.sparse.linalg as spla
from pycutfem.utils.meshgen import structured_quad
from pycutfem.core import Mesh
from pycutfem.assembly import stiffness_matrix, assemble
from pycutfem.assembly.load_vector import element_load
from pycutfem.assembly.boundary_conditions import apply_dirichlet
from pycutfem.io import plot_mesh

u_exact = lambda x,y: x**2*y + np.sin(y)
f_rhs   = lambda x,y: -2*y + np.sin(y)

nodes, elems = structured_quad(3,2, nx=30, ny=20)
mesh = Mesh(nodes, elems, 'quad')
KeFe = lambda eid: stiffness_matrix(mesh, eid)
K = assemble(mesh, lambda eid: stiffness_matrix(mesh, eid))
F = np.zeros(len(nodes))
for eid,elem in enumerate(mesh.elements):
    Fe = element_load(mesh, eid, f_rhs)
    for a,A in enumerate(elem): F[A]+=Fe[a]
dbc={}
for dof,(x,y) in enumerate(mesh.nodes):
    if np.isclose(x,0) or np.isclose(x,3) or np.isclose(y,0) or np.isclose(y,2):
        dbc[dof]=u_exact(x,y)
K,F = apply_dirichlet(K,F,dbc)
uh = spla.spsolve(K,F)
print('L2 error =', np.linalg.norm(uh - u_exact(nodes[:,0], nodes[:,1]),2)/len(nodes))
plot_mesh(mesh)
