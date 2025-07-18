import numpy as np
import types, sys
mod = types.ModuleType('vis')
mod.visualize_mesh_node_order = lambda *a, **k: None
mod.plot_mesh = lambda *a, **k: None
sys.modules.setdefault('pycutfem.io.visualization', mod)
mpl = types.ModuleType('mpl')
mpl.pyplot = types.ModuleType('plt')
mpl.tri = types.ModuleType('tri')
mpl.colors = types.ModuleType('colors')
mpl.animation = types.ModuleType('anim')
class DummyCmap:
    @staticmethod
    def from_list(*args, **kwargs):
        return None
mpl.colors.LinearSegmentedColormap = DummyCmap
mpl.animation.FuncAnimation = object
sys.modules.setdefault('matplotlib', mpl)
sys.modules.setdefault('matplotlib.pyplot', mpl.pyplot)
sys.modules.setdefault('matplotlib.tri', mpl.tri)
sys.modules.setdefault('matplotlib.colors', mpl.colors)
sys.modules.setdefault('matplotlib.animation', mpl.animation)
import scipy.sparse.linalg as spla
from pycutfem.core.mesh import Mesh
from pycutfem.core.topology import Node
from pycutfem.core.levelset import LevelSetFunction
from pycutfem.utils.domain_manager import get_domain_bitset
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.expressions import (
    Function, TrialFunction, TestFunction, grad, inner, Jump, dot, Constant, FacetNormal
)
from pycutfem.ufl.measures import dx, dInterface, dGhost
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.solvers.nonlinear_solver import NewtonSolver, NewtonParameters, TimeStepperParameters

def simple_structured_quad(nx, ny, order=1, Lx=1.0, Ly=1.0):
    num_x = order * nx + 1
    num_y = order * ny + 1
    xs = np.linspace(0, Lx, num_x)
    ys = np.linspace(0, Ly, num_y)
    nodes = []
    nid_counter = 0
    for y in ys:
        for x in xs:
            nodes.append(Node(nid_counter, x, y))
            nid_counter += 1
    elements = []
    corners = []
    def nid(ix, iy):
        return iy * num_x + ix
    for j in range(ny):
        for i in range(nx):
            bl = nid(i*order, j*order)
            br = nid(i*order+order, j*order)
            tr = nid(i*order+order, j*order+order)
            tl = nid(i*order, j*order+order)
            corners.append([bl, br, tr, tl])
            elem = []
            for jj in range(order+1):
                for ii in range(order+1):
                    elem.append(nid(i*order+ii, j*order+jj))
            elements.append(elem)
    return nodes, np.array(elements), np.empty((0,2),dtype=int), np.array(corners)

class LineLevelSet(LevelSetFunction):
    def __call__(self, x):
        x = np.asarray(x)
        return x[...,0]-0.5
    def gradient(self, x):
        g = np.array([1.0,0.0])
        return g if x.ndim==1 else np.tile(g,(x.shape[0],1))


def test_newton_volume_interface_ghost():
    poly=1
    nodes, elems, _, corners = simple_structured_quad(4,4,order=poly)
    mesh = Mesh(nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type='quad', poly_order=poly)
    ls = LineLevelSet()
    mesh.classify_elements(ls)
    mesh.classify_edges(ls)
    neg = get_domain_bitset(mesh,'element','outside')
    pos = get_domain_bitset(mesh,'element','inside')
    cut = get_domain_bitset(mesh,'element','cut')
    ghost = get_domain_bitset(mesh,'edge','ghost')
    me = MixedElement(mesh, field_specs={'u_neg':poly,'u_pos':poly})
    dh = DofHandler(me, method='cg')
    u_n = Function('u_neg','u_neg',dh)
    u_p = Function('u_pos','u_pos',dh)
    v_n = TestFunction('u_neg',dof_handler=dh)
    v_p = TestFunction('u_pos',dof_handler=dh)
    du_n = TrialFunction('u_neg',dof_handler=dh)
    du_p = TrialFunction('u_pos',dof_handler=dh)
    normal = FacetNormal()
    jump_u = Jump(u_p, u_n)
    jump_v = Jump(v_p, v_n)
    avg_flux_u = -0.5*(dot(grad(u_n),normal)+dot(grad(u_p),normal))
    avg_flux_v = -0.5*(dot(grad(v_n),normal)+dot(grad(v_p),normal))
    a = inner(grad(du_n),grad(v_n))*dx(defined_on=neg|cut)
    a += inner(grad(du_p),grad(v_p))*dx(defined_on=pos|cut)
    a += (dot(avg_flux_u,jump_v)+dot(avg_flux_v,jump_u)+10*jump_u*jump_v)*dInterface(level_set=ls)
    a += Jump(du_n)*Jump(v_n)*dGhost(defined_on=ghost,level_set=ls)
    a += Jump(du_p)*Jump(v_p)*dGhost(defined_on=ghost,level_set=ls)
    r = inner(grad(u_n),grad(v_n))*dx(defined_on=neg|cut)
    r += inner(grad(u_p),grad(v_p))*dx(defined_on=pos|cut)
    r += (dot(avg_flux_u,jump_v)+dot(avg_flux_v,jump_u)+10*jump_u*jump_v)*dInterface(level_set=ls)
    r += Jump(u_n)*Jump(v_n)*dGhost(defined_on=ghost,level_set=ls)
    r += Jump(u_p)*Jump(v_p)*dGhost(defined_on=ghost,level_set=ls)
    r -= Constant(1.0)*v_n*dx(defined_on=neg|cut)
    r -= Constant(1.0)*v_p*dx(defined_on=pos|cut)
    bcs = [BoundaryCondition('u_neg','dirichlet','boundary',0.0),
           BoundaryCondition('u_pos','dirichlet','boundary',0.0)]
    solver = NewtonSolver(r, a, dof_handler=dh, mixed_element=me, bcs=bcs, bcs_homog=bcs,
                          newton_params=NewtonParameters(newton_tol=1e-10, max_newton_iter=4),
                          quad_order=poly+2)
    delta, steps, _ = solver.solve_time_interval(functions=[u_n,u_p], prev_functions=[Function('u_neg_prev','u_neg',dh),Function('u_pos_prev','u_pos',dh)], time_params=TimeStepperParameters(max_steps=1))
    assert np.linalg.norm(delta, np.inf) < 1e-8
