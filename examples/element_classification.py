from pycutfem.utils import delaunay_rectangle
from pycutfem.io import plot_mesh
from pycutfem.cutters import classify_elements, classify_edges
from pycutfem.core import Mesh, CircleLevelSet
import numpy as np

nodes, elems = delaunay_rectangle(3.0, 2.0, nx=40, ny=40)
mesh = Mesh(nodes, elems, element_type='tri')
phi  = CircleLevelSet(center=(0.6, 0.5), radius=0.3)

classify_elements(mesh, phi)
classify_edges(mesh, phi)

plot_mesh(mesh, level_set=phi, edge_colors=True)
