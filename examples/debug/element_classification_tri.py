from pycutfem.utils.meshgen import structured_triangles
from pycutfem.io import plot_mesh
from pycutfem.cutters import classify_elements, classify_edges
from pycutfem.core import Mesh, CircleLevelSet
import numpy as np

poly_order = 2
nodes, elems = structured_triangles(3.0, 2.0, 40, 40, poly_order=poly_order)
mesh = Mesh(nodes, elems, element_type='tri', poly_order=poly_order)
phi  = CircleLevelSet(center=(0.6, 0.5), radius=0.3)

classify_elements(mesh, phi)
classify_edges(mesh, phi)

plot_mesh(mesh, level_set=phi, edge_colors=True)
