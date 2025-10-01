# -*- coding: utf-8 -*-
"""
Refined term-by-term comparison between PyCutFEM and NGSolve/XFEM.

This version has been refactored to compare the assembled matrices directly,
independent of any analytical solution. It achieves this by calculating the
action of the assembled operator K on vectors of all ones (u_vec, v_vec),
effectively comparing the sum of all elements in the matrices. This provides
a "fingerprint" to verify that both libraries generate the same discrete operator.

This version includes a comprehensive and granular set of tests to isolate
specific operators (mass, stiffness, etc.) and regions (pos, neg, bulk, interface).

Run:
    python compare_with_ngsolve.py
"""

import numpy as np
import scipy.sparse.linalg as spla
from dataclasses import dataclass
from typing import Optional

# ---------- PyCutFEM imports ----------
from pycutfem.core.mesh import Mesh as pycutfem_Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad, _structured_pk, structured_triangles
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    VectorTrialFunction, VectorTestFunction, TrialFunction, TestFunction,
    Function, VectorFunction, Constant,
    grad as pc_grad, inner as pc_inner, div as pc_div, dot as pc_dot, jump as pc_jump,
    Pos, Neg, CellDiameter,
)
from pycutfem.ufl.measures import dx as pycutfem_dx, dInterface as pycutfem_dInterface
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.core.levelset import CircleLevelSet, LevelSetGridFunction, LevelSetMeshAdaptation as PCLevelSetMeshAdaptation
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.visualization import plot_mesh_2, add_measure_area_overlay, add_element_fill, add_element_outline, zoom_to_elements
import matplotlib.pyplot as plt
from pycutfem.utils.area_checks import per_element_circle_split_structured_quad, per_element_circle_split_structured_triangle
# ---------- NGSolve / XFEM imports ----------
from netgen.geom2d import SplineGeometry
from ngsolve import *
from xfem import *
from xfem.lsetcurv import *
import time
import os


@dataclass
class TestResult:
    description: str
    error: float
    passed: bool
    details: str
# ---------- Pretty printing utilities ----------
def srelerr(a, b, rtol=1e-9, atol=1e-12):
    return abs(a-b)

def verdict(err, tol=1e-8):
    ok = err <= tol
    mark = "\x1b[32m✓\x1b[0m" if ok else "\x1b[31m✗\x1b[0m"
    return ok, mark

def term_header(title):
    print(f"\n=== {title} ===")

def numerics_vs_analytic_by_elem(dh, level_set, R, center=(0.0, 0.0), q=6, print_top=10):
    """
    Augments each element row with numeric Apos/Aneg from the Python assembler,
    adds d_pos/d_neg and a 'score', and returns rows sorted by score (desc).
    """
    from pycutfem.ufl.measures import dx
    # if integrate_pc_constant_dx isn't in scope here, import it too:
    # from your_module import integrate_pc_constant_dx
    ONE = Constant(1.0)

    mesh = dh.mixed_element.mesh
    if mesh.element_type == "quad":
        rows = per_element_circle_split_structured_quad(mesh, R, center=center)
    elif mesh.element_type == "tri":
        rows = per_element_circle_split_structured_triangle(mesh, R, center=center)
    else:
        raise KeyError(f"Unsupported element_type '{mesh.element_type}'")

    for r in rows:
        eid = int(r['eid'])
        # POS (= outside), NEG (= inside) on this single element
        dx_pos = dx(defined_on=[eid], level_set=level_set, metadata={'side': '+', 'q': q})
        dx_neg = dx(defined_on=[eid], level_set=level_set, metadata={'side': '-', 'q': q})

        Apos_num = float(integrate_pc_constant_dx(dh, ONE, dx_pos))
        Aneg_num = float(integrate_pc_constant_dx(dh, ONE, dx_neg))

        r['Apos_num'] = Apos_num
        r['Aneg_num'] = Aneg_num
        r['d_pos']    = Apos_num - r['Apos']
        r['d_neg']    = Aneg_num - r['Aneg']
        r['score']    = abs(r['d_pos']) + abs(r['d_neg'])

    rows_sorted = sorted(rows, key=lambda r: r['score'], reverse=True)

    if print_top:
        print("\nTop element area mismatches (analytic vs assembled):")
        for r in rows_sorted[:print_top]:
            print(
                f"eid={r['eid']:4d}  "
                f"Aneg ana/num={r['Aneg']:.8f}/{r['Aneg_num']:.8f}  "
                f"Apos ana/num={r['Apos']:.8f}/{r['Apos_num']:.8f}  "
                f"d_neg={r['d_neg']:+.2e}  d_pos={r['d_pos']:+.2e}  "
                f"score={r['score']:.2e}"
            )

    return rows_sorted



# ---------- PyCutFEM Setup ----------
def setup_pc(maxh=0.125, order=2, R=2.0/3.0, L = 2.0, H = 2.0, level_set_order = 1, use_quad = False):
    """
    Prepares all necessary PyCutFEM components (mesh, spaces, functions, measures)
    and returns them in a dictionary without assembling any forms.
    """
    nx = int(L / maxh)
    geo_order = max(1, order)
    if use_quad:
        nodes, elems, _, corners = structured_quad(
            L, H, nx=nx, ny=nx, poly_order=geo_order, offset=[-L/2, -H/2]
        )
        mesh = pycutfem_Mesh(nodes, elems, elements_corner_nodes=corners,
                            element_type="quad", poly_order=geo_order)
    else:
        nodes, elems, _, corners = structured_triangles(
            L, H, nx_quads=nx, ny_quads=nx, poly_order=geo_order, offset=[-L/2, -H/2]
        )
        mesh = pycutfem_Mesh(nodes, elems, elements_corner_nodes=corners,
                            element_type="tri", poly_order=geo_order)
    analytical_level_set = CircleLevelSet(center=(0.0, 0.0), radius=R)

    adapter = PCLevelSetMeshAdaptation(mesh, order=order, threshold=10.5)
    deformation = adapter.calc_deformation(analytical_level_set)
    lset_p1 = adapter.lset_p1

    me = MixedElement(mesh, field_specs={
        'u_pos_x': order, 'u_pos_y': order, 'p_pos_': order-1,
        'u_neg_x': order, 'u_neg_y': order, 'p_neg_': order-1,
        'lm': ':number:',
        'phi': level_set_order # level set P1
    })
    dh = DofHandler(me, method='cg')
    ls = LevelSetGridFunction(dh, field='phi')
    ls.interpolate(lambda x,y: np.hypot(x,y) - R)  # initial condition
    ls.commit()                                     # classify & build segments
    mesh.classify_elements(lset_p1)
    mesh.classify_edges(lset_p1)
    mesh.build_interface_segments(lset_p1)
    used_level_set = lset_p1


    inside, outside, interface = (
        mesh.element_bitset("inside"),
        mesh.element_bitset("outside"),
        mesh.element_bitset("cut")
    )
    has_inside = inside | interface
    has_outside = outside | interface
    

    Vpos = FunctionSpace("Vpos", ['u_pos_x','u_pos_y'], side='+')
    Vneg = FunctionSpace("Vneg", ['u_neg_x','u_neg_y'], side='-')
    
    qvol = 2*order + 4
    es = {
        'has_outside': has_outside, 'has_inside': has_inside,
        'inside': inside, 'outside': outside, 'interface': interface
    }
    
    # Pack and return all components
    return {
        'dh': dh, 'qvol': qvol, 'ONE': Constant(1.0),
        'up': VectorTrialFunction(Vpos, dh, side='+'), 'vp': VectorTestFunction(Vpos, dh, side='+'),
        'un': VectorTrialFunction(Vneg, dh, side='-'), 'vn': VectorTestFunction(Vneg, dh, side='-'),
        'pp': TrialFunction('p_pos_', dh, side='+'), 'qp': TestFunction('p_pos_', dh, side='+'),
        'pn': TrialFunction('p_neg_', dh, side='-'), 'qn': TestFunction('p_neg_', dh, side='-'),
        'nL': TrialFunction(name='nL', field_name='lm', dof_handler=dh), 'mL': TestFunction(name='mL', field_name='lm', dof_handler=dh),
        # 'nL': Function(name='nL', field_name='lm', dof_handler=dh), 'mL': Function(name='mL', field_name='lm', dof_handler=dh),
        'es': es,
        'level_set': used_level_set,
        'deformation': deformation,
        'dGamma': pycutfem_dInterface(defined_on=interface, level_set=used_level_set, metadata={'q': qvol}),
        'mu0': Constant(1.0), 'mu1': Constant(10.0), 'lam_val': 0.5*(1.0+10.0)*20*order**2,
    }


# ---------- NGSolve Setup ----------
def setup_ng(maxh=0.125, order=2, R=2.0/3.0, no_dirichlet=True, L = 2.0, H = 2.0,level_set_order = 1, use_quad = False):
    """
    Prepares all necessary NGSolve components and returns them in a dictionary.
    """
    square = SplineGeometry()
    square.AddRectangle((-L/2.0, -H/2.0), (L/2.0, H/2.0), bcs=[1, 2, 3, 4])
    if use_quad:
        mesh = Mesh(square.GenerateMesh(maxh=maxh, quad_dominated=True))
    else:
        mesh = Mesh(square.GenerateMesh(maxh=maxh, quad_dominated=False))

    # Level-set
    levelset = sqrt(x**2 + y**2) - R
    lsetp1 = GridFunction(H1(mesh,order=level_set_order))
    InterpolateToP1(levelset, lsetp1)
    ci = CutInfo(mesh, lsetp1)

    # Function spaces
    Vhbase = VectorH1(mesh, order=order) if no_dirichlet else VectorH1(mesh, order=order, dirichlet=[1,2,3,4])
    Qhbase = H1(mesh, order=order-1)
    Vhneg = Compress(Vhbase, GetDofsOfElements(Vhbase, ci.GetElementsOfType(HASNEG)))
    Vhpos = Compress(Vhbase, GetDofsOfElements(Vhbase, ci.GetElementsOfType(HASPOS)))
    Qhneg = Compress(Qhbase, GetDofsOfElements(Qhbase, ci.GetElementsOfType(HASNEG)))
    Qhpos = Compress(Qhbase, GetDofsOfElements(Qhbase, ci.GetElementsOfType(HASPOS)))
    WhG = FESpace([Vhneg*Vhpos, Qhneg*Qhpos, NumberSpace(mesh)], dgjumps=True)
    # WhG = FESpace([Vhbase*Vhbase, Qhbase*Qhbase, NumberSpace(mesh)], dgjumps=True)

    # Trial and Test functions
    u, p, n = WhG.TrialFunction()
    v, q, m = WhG.TestFunction()
    
    # GridFunctions for holding solution vectors (will be filled with ones)
    gfu = GridFunction(WhG)
    gfv = GridFunction(WhG)
        
    mu0, mu1 = 1.0, 10.0

    return {
        'mesh': mesh, 'WhG': WhG, 'gfu': gfu, 'gfv': gfv,
        'u': u, 'p': p, 'n': n, 'v': v, 'q': q, 'm': m,
        "ci": ci, 'lsetp1': lsetp1,
        'dx_neg_all': dCut(lsetp1, NEG), 'dx_pos_all': dCut(lsetp1, POS), 'ds': dCut(lsetp1, IF),
        'dx_neg_bulk': dCut(lsetp1, NEG, definedonelements=ci.GetElementsOfType(NEG)),
        'dx_pos_bulk': dCut(lsetp1, POS, definedonelements=ci.GetElementsOfType(POS)),
        'dx_neg_iface': dCut(lsetp1, NEG, definedonelements=ci.GetElementsOfType(IF)),
        'dx_pos_iface': dCut(lsetp1, POS, definedonelements=ci.GetElementsOfType(IF)),
        'mu0': mu0, 'mu1': mu1, 'lam_val': 0.5*(mu0+mu1)*20*order*order,
    }

def setup_ng_deformation(maxh=0.125, order=2, R=2.0/3.0, no_dirichlet=True, L = 2.0, H = 2.0,level_set_order = 1, use_quad = False):
    """
    Prepares all necessary NGSolve components and returns them in a dictionary.
    """
    square = SplineGeometry()
    square.AddRectangle((-L/2.0, -H/2.0), (L/2.0, H/2.0), bcs=[1, 2, 3, 4])
    if use_quad:
        mesh = Mesh(square.GenerateMesh(maxh=maxh, quad_dominated=True))
    else:
        mesh = Mesh(square.GenerateMesh(maxh=maxh, quad_dominated=False))

    # Level-set
    levelset = sqrt(x**2 + y**2) - R
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order, threshold=10.5,
                                      discontinuous_qn=True)
    deformation = lsetmeshadap.CalcDeformation(levelset)
    lsetp1 = lsetmeshadap.lset_p1
    ci = CutInfo(mesh, lsetp1)

    # Function spaces
    Vhbase = VectorH1(mesh, order=order) if no_dirichlet else VectorH1(mesh, order=order, dirichlet=[1,2,3,4])
    Qhbase = H1(mesh, order=order-1)
    Vhneg = Compress(Vhbase, GetDofsOfElements(Vhbase, ci.GetElementsOfType(HASNEG)))
    Vhpos = Compress(Vhbase, GetDofsOfElements(Vhbase, ci.GetElementsOfType(HASPOS)))
    Qhneg = Compress(Qhbase, GetDofsOfElements(Qhbase, ci.GetElementsOfType(HASNEG)))
    Qhpos = Compress(Qhbase, GetDofsOfElements(Qhbase, ci.GetElementsOfType(HASPOS)))
    WhG = FESpace([Vhneg*Vhpos, Qhneg*Qhpos, NumberSpace(mesh)], dgjumps=True)
    # WhG = FESpace([Vhbase*Vhbase, Qhbase*Qhbase, NumberSpace(mesh)], dgjumps=True)

    # Trial and Test functions
    u, p, n = WhG.TrialFunction()
    v, q, m = WhG.TestFunction()
    
    # GridFunctions for holding solution vectors (will be filled with ones)
    gfu = GridFunction(WhG)
    gfv = GridFunction(WhG)
        
    mu0, mu1 = 1.0, 10.0

    return {
        'mesh': mesh, 'WhG': WhG, 'gfu': gfu, 'gfv': gfv,
        'u': u, 'p': p, 'n': n, 'v': v, 'q': q, 'm': m,
        "ci": ci, 'lsetp1': lsetp1,
        'dx_neg_all': dCut(lsetp1, NEG, deformation=deformation), 
        'dx_pos_all': dCut(lsetp1, POS, deformation=deformation), 'ds': dCut(lsetp1, IF, deformation=deformation),
        'dx_neg_bulk': dCut(lsetp1, NEG, definedonelements=ci.GetElementsOfType(NEG), deformation=deformation),
        'dx_pos_bulk': dCut(lsetp1, POS, definedonelements=ci.GetElementsOfType(POS), deformation=deformation),
        'dx_neg_iface': dCut(lsetp1, NEG, definedonelements=ci.GetElementsOfType(IF), deformation=deformation),
        'dx_pos_iface': dCut(lsetp1, POS, definedonelements=ci.GetElementsOfType(IF), deformation=deformation),
        'mu0': mu0, 'mu1': mu1, 'lam_val': 0.5*(mu0+mu1)*20*order*order,
        'deformation': deformation
    }




# ---------- Assembly and Energy Calculation Wrappers ----------
def assemble_and_energy_ng(bf: BilinearForm, gfu, gfv):
    with TaskManager():
        bf.Assemble()
    tmp = gfv.vec.CreateVector()
    bf.Apply(gfu.vec, tmp)
    return InnerProduct(gfv.vec, tmp)

def assemble_and_energy_pc(form, dh, u_vec, v_vec, backend='python'):
    K, _ = assemble_form(Equation(form, None), dof_handler=dh, quad_order=None, backend=backend)
    return float(v_vec @ (K @ u_vec))
def integrate_cf_dx(mesh, cf, dx_measure):
    """
    Integrate scalar CoefficientFunction 'cf' over a measure by assembling
    a LinearForm on NumberSpace with 'cf * w * measure'.
    """
    NS = NumberSpace(mesh)
    w = NS.TestFunction()
    lf = LinearForm(NS)
    lf += cf * w * dx_measure
    with TaskManager():
        lf.Assemble()
    return float(lf.vec.FV()[0])
def integrate_pc_constant_dx(dh, form, dx_measure, backend='python'):
    w = TestFunction(field_name='lm', dof_handler=dh)
    L = form * w * dx_measure
    _, F = assemble_form(Equation(None, L), dof_handler=dh, quad_order=None, backend=backend)
    lm_inds = dh.get_field_slice('lm')         # returns [global_index_of_lm]
    return float(F[lm_inds][0])

class PC_DX:
    """A manager for PyCutFEM integration measures."""
    def __init__(self, quad_order, level_set, element_sets, deformation=None):
        self.quad_order = quad_order
        self.level_set = level_set
        self.es = element_sets # Dictionary of element bitsets
        self.deformation = deformation

    def pos_all(self, quad_order=None):
        if quad_order is None:
            quad_order = self.quad_order
        return pycutfem_dx(defined_on=self.es['has_outside'], level_set=self.level_set,
                           metadata={'side': '+', 'q': quad_order}, deformation=self.deformation)

    def neg_all(self, quad_order=None):
        if quad_order is None:
            quad_order = self.quad_order
        return pycutfem_dx(defined_on=self.es['has_inside'], level_set=self.level_set,
                           metadata={'side': '-', 'q': quad_order}, deformation=self.deformation)

    def pos_bulk(self, quad_order=None):
        if quad_order is None:
            quad_order = self.quad_order
        return pycutfem_dx(defined_on=self.es['outside'], level_set=self.level_set,
                           metadata={'side': '+', 'q': quad_order}, deformation=self.deformation)

    def neg_bulk(self, quad_order=None):
        if quad_order is None:
            quad_order = self.quad_order
        return pycutfem_dx(defined_on=self.es['inside'], level_set=self.level_set,
                           metadata={'side': '-', 'q': quad_order}, deformation=self.deformation) 

    def pos_iface(self, quad_order=None):
        if quad_order is None:
            quad_order = self.quad_order
        return pycutfem_dx(defined_on=self.es['interface'], level_set=self.level_set,
                           metadata={'side': '+', 'q': quad_order}, deformation=self.deformation)

    def neg_iface(self, quad_order=None):
        if quad_order is None:
            quad_order = self.quad_order
        return pycutfem_dx(defined_on=self.es['interface'], level_set=self.level_set,
                           metadata={'side': '-', 'q': quad_order}, deformation=self.deformation)

    def dGamma(self, quad_order=None, side = None): # Corrected name from dGamama
        if quad_order is None:
            quad_order = self.quad_order
        return pycutfem_dInterface(defined_on=self.es['interface'], level_set=self.level_set,
                                    metadata={'q': quad_order, 'side': side}, deformation=self.deformation)

class NG_DX:
    """A manager for NGSolve/XFEM integration measures."""
    def __init__(self, quad_order, lsetp1, cut_info, deformation= None):
        self.quad_order = quad_order
        self.lsetp1 = lsetp1
        self.ci = cut_info
        self.deformation = deformation

    def pos_all(self, quad_order=None):
        if quad_order is None:
            quad_order = self.quad_order
        if self.deformation is None:
            return dCut(self.lsetp1, POS, definedonelements=self.ci.GetElementsOfType(HASPOS), order=quad_order)
        return dCut(self.lsetp1, POS, definedonelements=self.ci.GetElementsOfType(HASPOS), order=quad_order, deformation=self.deformation)

    def neg_all(self, quad_order=None):
        if quad_order is None:
            quad_order = self.quad_order
        if self.deformation is None:
            return dCut(self.lsetp1, NEG, definedonelements=self.ci.GetElementsOfType(HASNEG), order=quad_order)
        return dCut(self.lsetp1, NEG, definedonelements=self.ci.GetElementsOfType(HASNEG), order=quad_order, deformation=self.deformation)

    def pos_bulk(self, quad_order=None):
        if quad_order is None:
            quad_order = self.quad_order
        if self.deformation is None:
            return dCut(self.lsetp1, POS, definedonelements=self.ci.GetElementsOfType(POS), order=quad_order)
        return dCut(self.lsetp1, POS, definedonelements=self.ci.GetElementsOfType(POS), order=quad_order, deformation=self.deformation)

    def neg_bulk(self, quad_order=None):
        if quad_order is None:
            quad_order = self.quad_order
        if self.deformation is None:
            return dCut(self.lsetp1, NEG, definedonelements=self.ci.GetElementsOfType(NEG), order=quad_order)
        return dCut(self.lsetp1, NEG, definedonelements=self.ci.GetElementsOfType(NEG), order=quad_order, deformation=self.deformation)

    def pos_iface(self, quad_order=None):
        if quad_order is None:
            quad_order = self.quad_order
        if self.deformation is None:
            return dCut(self.lsetp1, POS, definedonelements=self.ci.GetElementsOfType(IF), order=quad_order)
        return dCut(self.lsetp1, POS, definedonelements=self.ci.GetElementsOfType(IF), order=quad_order, deformation=self.deformation)

    def neg_iface(self, quad_order=None):
        if quad_order is None:
            quad_order = self.quad_order
        if self.deformation is None:
            return dCut(self.lsetp1, NEG, definedonelements=self.ci.GetElementsOfType(IF), order=quad_order)
        return dCut(self.lsetp1, NEG, definedonelements=self.ci.GetElementsOfType(IF), order=quad_order, deformation=self.deformation)

    def dGamma(self, quad_order=None): # Corrected name
        if quad_order is None:
            quad_order = self.quad_order
        if self.deformation is not None:
            return dCut(self.lsetp1, IF, definedonelements=self.ci.GetElementsOfType(IF), order=quad_order, deformation=self.deformation)
        return dCut(self.lsetp1, IF, definedonelements=self.ci.GetElementsOfType(IF), order=quad_order)
# ---------- Runner ----------
def main(backend='python'):
    maxh, order, R = 0.125, 2, 2.0/3.0
    L, H = 2.0, 2.0
    use_quad = True  # Whether to use quad elements in PyCutFEM and NGSolve
    
    # --- 1. Setup Phase ---
    t0 = time.time()
    print("Setting up PyCutFEM and NGSolve problems"
          f" with maxh={maxh}, order={order}, R={R}, use_quad={use_quad} with backend={backend}...")
    pc_setup = setup_pc(maxh, order, R, L, H, use_quad=use_quad)
    ng_setup = setup_ng(maxh, order, R, L, H, use_quad=use_quad)
    ng_setup_def = setup_ng_deformation(maxh, order, R, L, H, use_quad=use_quad)
    quad_order = 8
    pc_dx = PC_DX(quad_order, pc_setup['level_set'], pc_setup['es'])
    pc_dx_def = PC_DX(quad_order, pc_setup['level_set'], pc_setup['es'], deformation=pc_setup['deformation'])
    ng_dx = NG_DX(quad_order, ng_setup['lsetp1'], ng_setup['ci'])
    ng_dx_def = NG_DX(quad_order, ng_setup_def['lsetp1'], ng_setup_def['ci'], deformation=ng_setup_def['deformation'])

    # --- Create vectors of all ONES for matrix comparison ---
    total_dofs_pc = pc_setup['dh'].total_dofs
    u_vec = np.ones(total_dofs_pc)
    v_vec = np.ones(total_dofs_pc)
    
    ng_setup['gfu'].vec[:] = 0.0
    ng_setup['gfv'].vec[:] = 0.0
    ng_setup_def['gfu'].vec[:] = 0.0
    ng_setup_def['gfv'].vec[:] = 0.0
    # Define the constant vector field u=v=(1,1)
    ONE = CoefficientFunction(1.0)
    const_vec_field = CoefficientFunction((1, 1))
    const_scalar_field = CoefficientFunction(1.0)
    with TaskManager():
        # Set velocity components (u, v)
        ng_setup['gfu'].components[0].components[1].Set(const_vec_field) # u_pos
        ng_setup['gfv'].components[0].components[1].Set(const_vec_field) # v_pos
        ng_setup['gfu'].components[0].components[0].Set(const_vec_field) # u_neg
        ng_setup['gfv'].components[0].components[0].Set(const_vec_field) # v_neg

        # Set pressure components (p, q)
        ng_setup['gfu'].components[1].components[1].Set(const_scalar_field) # p_pos
        ng_setup['gfv'].components[1].components[1].Set(const_scalar_field) # q_pos
        ng_setup['gfu'].components[1].components[0].Set(const_scalar_field) # p_neg
        ng_setup['gfv'].components[1].components[0].Set(const_scalar_field) # q_neg

        # Set number space components
        ng_setup['gfu'].components[2].Set(CoefficientFunction(1.0)) # lm for u
        ng_setup['gfv'].components[2].Set(CoefficientFunction(1.0)) # lm for v
    
    with TaskManager():
        # Set velocity components (u, v)
        ng_setup_def['gfu'].components[0].components[1].Set(const_vec_field) # u_pos
        ng_setup_def['gfv'].components[0].components[1].Set(const_vec_field) # v_pos
        ng_setup_def['gfu'].components[0].components[0].Set(const_vec_field) # u_neg
        ng_setup_def['gfv'].components[0].components[0].Set(const_vec_field) # v_neg

        # Set pressure components (p, q)
        ng_setup_def['gfu'].components[1].components[1].Set(const_scalar_field) # p_pos
        ng_setup_def['gfv'].components[1].components[1].Set(const_scalar_field) # q_pos
        ng_setup_def['gfu'].components[1].components[0].Set(const_scalar_field) # p_neg
        ng_setup_def['gfv'].components[1].components[0].Set(const_scalar_field) # q_neg

        # Set number space components
        ng_setup_def['gfu'].components[2].Set(CoefficientFunction(1.0)) # lm for u
        ng_setup_def['gfv'].components[2].Set(CoefficientFunction(1.0)) # lm for v
    all_tests = [] # For summary

    # Helper UFL functions
    def eps_pc(u): return 0.5*(pc_grad(u)+pc_grad(u).T)
    def eps_ng(u): return 0.5*(Grad(u)+Grad(u).trans)


    area_pos_pc = integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], pc_dx.pos_all(), backend=backend) 
    area_pos_pc_def = integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], pc_dx_def.pos_all(), backend=backend) 
    area_neg_pc = integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], pc_dx.neg_all(), backend=backend)
    area_neg_pc_def = integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], pc_dx_def.neg_all(), backend=backend)
    
    area_pos_ng =  integrate_cf_dx(ng_setup['mesh'], ONE, ng_dx.pos_all())
    area_neg_ng =  integrate_cf_dx(ng_setup['mesh'], ONE, ng_dx.neg_all())
    area_pos_ng_def =  integrate_cf_dx(ng_setup_def['mesh'], ONE, ng_dx_def.pos_all())
    area_neg_ng_def =  integrate_cf_dx(ng_setup_def['mesh'], ONE, ng_dx_def.neg_all())
    area_combined_pc = area_pos_pc + area_neg_pc
    area_combined_pc_def = area_pos_pc_def + area_neg_pc_def
    area_combined_ng = area_pos_ng + area_neg_ng
    area_combined_ng_def = area_pos_ng_def + area_neg_ng_def
    total_exact_area = L * H
    exact_neg_area = np.pi * R**2
    exact_pos_area = total_exact_area - exact_neg_area
    print(f"PC Areas comparison: +ve Area diff: {area_pos_pc - exact_pos_area:+.2e}, -ve Area diff: {area_neg_pc - exact_neg_area:+.2e}, Combined Area diff: {area_combined_pc - total_exact_area:+.2e}")
    print(f"PC (with deformation) Areas comparison: +ve Area diff: {area_pos_pc_def - exact_pos_area:+.2e}, -ve Area diff: {area_neg_pc_def - exact_neg_area:+.2e}, Combined Area diff: {area_combined_pc_def - total_exact_area:+.2e}")
    
    print(f"NG Areas comparison: +ve Area diff: {area_pos_ng - exact_pos_area:+.2e}, -ve Area diff: {area_neg_ng - exact_neg_area:+.2e}, Combined Area diff: {area_combined_ng - total_exact_area:+.2e}")
    print(f"NG (with deformation) Areas comparison: +ve Area diff: {area_pos_ng_def - exact_pos_area:+.2e}, -ve Area diff: {area_neg_ng_def - exact_neg_area:+.2e}, Combined Area diff: {area_combined_ng_def - total_exact_area:+.2e}")
    err_A_pos = srelerr(area_pos_pc, exact_pos_area)
    err_A_neg = srelerr(area_neg_pc, exact_neg_area)
    err_A_pos_def = srelerr(area_pos_pc_def, exact_pos_area)
    err_A_neg_def = srelerr(area_neg_pc_def, exact_neg_area)
    
    print(f"Relative error (Area +ve): {err_A_pos:+.2e}, (Area -ve): {err_A_neg:+.2e}")
    print(f"Relative error with deformation (Area +ve): {err_A_pos_def:+.2e}, (Area -ve): {err_A_neg_def:+.2e}")
    print(f"Relative error PC (Area -ve): {err_A_neg:+.2e}, NG (Area -ve): {srelerr(area_neg_ng, exact_neg_area):+.2e}, NG with deformation (Area -ve): {srelerr(area_neg_ng_def, exact_neg_area):+.2e}")
    print(f"Relative error PC with deformation (Area -ve): {err_A_neg_def:+.2e}, NG (Area -ve): {srelerr(area_neg_ng, exact_neg_area):+.2e}, NG with deformation (Area -ve): {srelerr(area_neg_ng_def, exact_neg_area):+.2e}")
    print(f"PC Areas: +ve {area_pos_pc:.8f}, -ve {area_neg_pc:.8f}, Combined {area_combined_pc:.8f}")
    print(f"PC (with deformation) Areas: +ve {area_pos_pc_def:.8f}, -ve {area_neg_pc_def:.8f}, Combined {area_combined_pc_def:.8f}")
    print(f"NG Areas: +ve {area_pos_ng:.8f}, -ve {area_neg_ng:.8f}, Combined {area_combined_ng:.8f}")
    print(f"Exact Areas: +ve {exact_pos_area:.8f}, -ve {exact_neg_area:.8f}, Combined {total_exact_area:.8f}")

    # Interface length check via PyCutFEM interface assembler (with/without deformation)
    exact_circ = 2.0 * np.pi * R
    dGamma_pc = pycutfem_dInterface(defined_on=pc_setup['es']['interface'], level_set=pc_setup['level_set'], metadata={'q': quad_order, 'profile': True})
    dGamma_pc_def = pycutfem_dInterface(defined_on=pc_setup['es']['interface'], level_set=pc_setup['level_set'], metadata={'q': quad_order, 'profile': True}, deformation=pc_setup.get('deformation', None))
    t1 = time.time()
    L_pc = integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], dGamma_pc, backend=backend)
    L_pc_def = integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], dGamma_pc_def, backend=backend)
    print("\nInterface Linear Length Checks (PC assembler):")
    print(f"Exact circumference: {exact_circ:.8f}")
    print(f"PC φ_P1 length     : {L_pc:.8f}, err={L_pc - exact_circ:+.6e}")
    print(f"PC deformed length : {L_pc_def:.8f}, err={L_pc_def - exact_circ:+.6e}")
    t2 = time.time()
    print(f"Time taken for interface: {t2 - t1:.4f} seconds")
    print(f"Total setup time: {t2 - t0:.4f} seconds")

    # ---------------- Pretty summary as a DataFrame -----------------
    try:
        import pandas as pd

        # Build rows: (sign, deformation)
        rows = [
            ("Area +", "No deformation",  area_pos_pc,     area_pos_ng,     exact_pos_area),
            ("Area +", "With deformation", area_pos_pc_def, area_pos_ng_def, exact_pos_area),
            ("Area -", "No deformation",  area_neg_pc,     area_neg_ng,     exact_neg_area),
            ("Area -", "With deformation", area_neg_pc_def, area_neg_ng_def, exact_neg_area),
        ]

        idx = pd.MultiIndex.from_tuples([(r[0], r[1]) for r in rows], names=["Region", "Mode"])
        cols = pd.MultiIndex.from_product([["PC", "NG"], ["value", "diff", "rel"]])

        data = []
        for (_, _, pc_val, ng_val, exact) in rows:
            pc_diff = pc_val - exact
            ng_diff = ng_val - exact
            pc_rel  = pc_diff / exact if exact != 0.0 else 0.0
            ng_rel  = ng_diff / exact if exact != 0.0 else 0.0
            data.append([
                pc_val, pc_diff, pc_rel,
                ng_val, ng_diff, ng_rel,
            ])

        df = pd.DataFrame(data, index=idx, columns=cols)

        # Print table with consistent formats
        def fmt_val(x: Optional[float]) -> str:
            return f"{x:.8f}" if isinstance(x, (float, int)) else f"{x}"
        def fmt_diff(x: Optional[float]) -> str:
            return f"{x:+.6e}" if isinstance(x, (float, int)) else f"{x}"
        def fmt_rel(x: Optional[float]) -> str:
            return f"{x:+.6e}" if isinstance(x, (float, int)) else f"{x}"

        fmt = {
            ("PC", "value"): fmt_val, ("PC", "diff"): fmt_diff, ("PC", "rel"): fmt_rel,
            ("NG", "value"): fmt_val, ("NG", "diff"): fmt_diff, ("NG", "rel"): fmt_rel,
        }

        print("\n== Area Error Summary (PC vs NG) ==")
        print(df.to_string(formatters=fmt))
    except Exception as e:
        print("\n[pandas unavailable] Area Error Summary (PC vs NG):")
        print("Region/Mode | PC(value, diff, rel) | NG(value, diff, rel)")
        for (reg, mode, pc_val, ng_val, exact) in rows:
            pc_diff = pc_val - exact
            ng_diff = ng_val - exact
            pc_rel  = pc_diff / exact if exact != 0.0 else 0.0
            ng_rel  = ng_diff / exact if exact != 0.0 else 0.0
            print(f"{reg:5s}/{mode:16s} | {pc_val:10.6f}, {pc_diff:+.3e}, {pc_rel:+.3e}"
                  f" | {ng_val:10.6f}, {ng_diff:+.3e}, {ng_rel:+.3e}")



    # --- 2. Central Test Case Definitions ---
    TEST_CASES = {
        # --- Mass Matrices (u,v) ---
        "mass_pos": {
            "description": "MASS (+) (u, v)", "type": "total",
            "pc_form": pc_inner(pc_setup['up'], pc_setup['vp']) * pc_dx.pos_all(),
            "ng_form": InnerProduct(ng_setup['u'][1], ng_setup['v'][1]) * ng_dx.pos_all(),
        },
        "mass_neg": {
            "description": "MASS (-) (u, v)", "type": "total",
            "pc_form": pc_inner(pc_setup['un'], pc_setup['vn']) * pc_dx.neg_all(),
            "ng_form": InnerProduct(ng_setup['u'][0], ng_setup['v'][0]) * ng_dx.neg_all(),
        },
        "mass_combined": {
            "description": "MASS (Combined) (u, v)", "type": "total",
            "pc_form": (pc_inner(pc_setup['up'], pc_setup['vp']) * pc_dx.pos_all() +
                        pc_inner(pc_setup['un'], pc_setup['vn']) * pc_dx.neg_all()),
            "ng_form": (InnerProduct(ng_setup['u'][1], ng_setup['v'][1]) * ng_dx.pos_all() +
                        InnerProduct(ng_setup['u'][0], ng_setup['v'][0]) * ng_dx.neg_all()),
        },
        "mass_scalar_pos": {
            "description": "MASS (Scalar) Positive (p, q)", "type": "total",
            "pc_form": (pc_setup['pp']*pc_setup['qp']*pc_dx.pos_all()  
                        ),
            "ng_form": (ng_setup['p'][1]*ng_setup['q'][1]*ng_dx.pos_all()
                        ),
        },
        "mass_scalar_neg": {
            "description": "MASS (Scalar) Negative (p, q)", "type": "total",
            "pc_form": (
                        pc_setup['pn']*pc_setup['qn']*pc_dx.neg_all()),
            "ng_form": (
                        ng_setup['p'][0]*ng_setup['q'][0]*ng_dx.neg_all()),
        },
        "mass_scalar_combined": {
            "description": "MASS (Scalar) Combined (p, q)", "type": "total",
            "pc_form": (pc_setup['pp']*pc_setup['qp']*pc_dx.pos_all() + 
                        pc_setup['pn']*pc_setup['qn']*pc_dx.neg_all()),
            "ng_form": (ng_setup['p'][1]*ng_setup['q'][1]*ng_dx.pos_all() + 
                        ng_setup['p'][0]*ng_setup['q'][0]*ng_dx.neg_all()),
        },

        # --- Stiffness/Laplacian Matrices (grad(u), grad(v)) ---
        "stiffness_pos": {
            "description": "STIFFNESS (+) (∇u:∇v)", "type": "total",
            "pc_form": pc_inner(pc_grad(pc_setup['up']), pc_grad(pc_setup['vp'])) * pc_dx.pos_all(),
            "ng_form": InnerProduct(Grad(ng_setup['u'][1]), Grad(ng_setup['v'][1])) * ng_dx.pos_all(),
        },
        "stiffness_neg": {
            "description": "STIFFNESS (-) (∇u:∇v)", "type": "total",
            "pc_form": pc_inner(pc_grad(pc_setup['un']), pc_grad(pc_setup['vn'])) * pc_dx.neg_all(),
            "ng_form": InnerProduct(Grad(ng_setup['u'][0]), Grad(ng_setup['v'][0])) * ng_dx.neg_all(),
        },

        # --- Full Volume/Elasticity Terms (eps(u), eps(v)) ---
        "volume_pos": {
            "description": "VOLUME (+) (2μ ε:ε)", "type": "total",
            "pc_form": 2*pc_setup['mu1']*pc_inner(eps_pc(pc_setup['up']), eps_pc(pc_setup['vp']))*pc_dx.pos_all(),
            "ng_form": 2*ng_setup['mu1']*InnerProduct(eps_ng(ng_setup['u'][1]), eps_ng(ng_setup['v'][1]))*ng_dx.pos_all(),
        },
        "volume_neg": {
            "description": "VOLUME (-) (2μ ε:ε)", "type": "total",
            "pc_form": 2*pc_setup['mu0']*pc_inner(eps_pc(pc_setup['un']), eps_pc(pc_setup['vn']))*pc_dx.neg_all(),
            "ng_form": 2*ng_setup['mu0']*InnerProduct(eps_ng(ng_setup['u'][0]), eps_ng(ng_setup['v'][0]))*ng_dx.neg_all(),
        },
        "volume_combined": {
            "description": "VOLUME (Combined) (2μ ε:ε)", "type": "total",
            "parent_of_split": "volume_split", # Link to the corresponding split test
            "pc_form": (2*pc_setup['mu1']*pc_inner(eps_pc(pc_setup['up']), eps_pc(pc_setup['vp']))*pc_dx.pos_all() + 
                        2*pc_setup['mu0']*pc_inner(eps_pc(pc_setup['un']), eps_pc(pc_setup['vn']))*pc_dx.neg_all()),
            "ng_form": (2*ng_setup['mu1']*InnerProduct(eps_ng(ng_setup['u'][1]), eps_ng(ng_setup['v'][1]))*ng_dx.pos_all() +
                        2*ng_setup['mu0']*InnerProduct(eps_ng(ng_setup['u'][0]), eps_ng(ng_setup['v'][0]))*ng_dx.neg_all()),
        },

        # --- Divergence/Pressure Terms ---
        "divu_q": {
            "description": "DIVU_Q (-div(u)·q)", "type": "total",
            "parent_of_split": "divu_q_split",
            "pc_form": (-pc_div(pc_setup['up'])*pc_setup['qp'])*pc_dx.pos_all() + (-pc_div(pc_setup['un'])*pc_setup['qn'])*pc_dx.neg_all(),
            "ng_form": (-div(ng_setup['u'][1])*ng_setup['q'][1])*ng_dx.pos_all() + (-div(ng_setup['u'][0])*ng_setup['q'][0])*ng_dx.neg_all(),
        },
        "divv_p": {
            "description": "DIVV_P (-div(v)·p)", "type": "total",
            "parent_of_split": "divv_p_split",
            "pc_form": (-pc_div(pc_setup['vp'])*pc_setup['pp'])*pc_dx.pos_all() + (-pc_div(pc_setup['vn'])*pc_setup['pn'])*pc_dx.neg_all(),
            "ng_form": (-div(ng_setup['v'][1])*ng_setup['p'][1])*ng_dx.pos_all() + (-div(ng_setup['v'][0])*ng_setup['p'][0])*ng_dx.neg_all(),
        },
        # --- Mean Terms ---
        "mean_split_pos_1": {
            "description": "MEAN (n *q dx_neg)", "type": "total",
            "pc_form": (pc_setup['nL']*Neg(pc_setup['qn']) )*pc_dx.neg_all(),
            "ng_form": (ng_setup['n']*ng_setup['q'][0] )*ng_dx.neg_all(),
        },
        "mean_split_pos_2": {
            "description": "MEAN (m * p dx_neg)", "type": "total",
            "pc_form": ( pc_setup['mL']*Neg(pc_setup['pn']))*pc_dx.neg_all(),
            "ng_form": ( ng_setup['m']*ng_setup['p'][0])*ng_dx.neg_all(),
        },
        "mean_neg": {
            "description": "MEAN (n *q dx_neg + m * p dx_neg)", "type": "total",
            "pc_form": (pc_setup['nL']*Neg(pc_setup['qn']) + pc_setup['mL']*Neg(pc_setup['pn']))*pc_dx.neg_all(),
            "ng_form": (ng_setup['n']*ng_setup['q'][0] + ng_setup['m']*ng_setup['p'][0])*ng_dx.neg_all(),
        },
        "mean_pos": {
            "description": "MEAN (n *q dx_pos + m * p dx_pos)", "type": "total",
            "pc_form": (pc_setup['nL']*Pos(pc_setup['qp']) + pc_setup['mL']*Pos(pc_setup['pp']))*pc_dx.pos_all(),
            "ng_form": (ng_setup['n']*ng_setup['q'][1] + ng_setup['m']*ng_setup['p'][1])*ng_dx.pos_all(),
        },
        "mean_comb": {
            "description": "MEAN (n *q dx_pos + m * p dx_pos + n *q dx_neg + m * p dx_neg)", "type": "total",
            "pc_form": ((pc_setup['nL']*Pos(pc_setup['qp']) + pc_setup['mL']*Pos(pc_setup['pp']))*pc_dx.pos_all()
                        + (pc_setup['nL']*Neg(pc_setup['qn']) + pc_setup['mL']*Neg(pc_setup['pn']))*pc_dx.neg_all()),
            "ng_form": ((ng_setup['n']*ng_setup['q'][1] + ng_setup['m']*ng_setup['p'][1])*ng_dx.pos_all()
                        + (ng_setup['n']*ng_setup['q'][0] + ng_setup['m']*ng_setup['p'][0])*ng_dx.neg_all()),
        },
        # --- Area Terms ---
        "area_pos": {
            "description": "AREA (+) (1 dx_has_pos)", "type": "direct",
            "pc_form": integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], pc_dx.pos_all(), backend=backend),
            "ng_form": integrate_cf_dx(ng_setup['mesh'], ONE, ng_dx.pos_all()),
        },
        "area_neg": {
            "description": "AREA (-) (1 dx_has_neg)", "type": "direct",
            "pc_form": integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], pc_dx.neg_all(), backend=backend),
            "ng_form": integrate_cf_dx(ng_setup['mesh'], ONE, ng_dx.neg_all()),
        },
        "area_combined": {
            "description": "AREA (Combined) (1 dx_has_pos + 1 dx_has_neg)", "type": "direct",
            "pc_form": (integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], pc_dx.pos_all(), backend=backend) +
                        integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], pc_dx.neg_all(), backend=backend)),
            "ng_form": (integrate_cf_dx(ng_setup['mesh'], ONE, ng_dx.pos_all()) +
                        integrate_cf_dx(ng_setup['mesh'], ONE, ng_dx.neg_all())),
        },
        "area_only_pos": {
            "description": "AREA (+) (1 dx_pos)", "type": "direct",
            "pc_form": integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], pc_dx.pos_bulk(), backend=backend),
            "ng_form": integrate_cf_dx(ng_setup['mesh'], ONE, ng_dx.pos_bulk()),
        },
        "area_only_neg": {
            "description": "AREA (-) (1 dx_neg)", "type": "direct",
            "pc_form": integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], pc_dx.neg_bulk(), backend=backend),
            "ng_form": integrate_cf_dx(ng_setup['mesh'], ONE, ng_dx.neg_bulk()),
        },
        "area_only_pos_neg": {
            "description": "AREA  (1 dx_neg) + (1 dx_pos)", "type": "direct",
            "pc_form": (integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], pc_dx.neg_bulk(), backend=backend) +
                        integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], pc_dx.pos_bulk(), backend=backend)),
            "ng_form": (integrate_cf_dx(ng_setup['mesh'], ONE, ng_dx.neg_bulk()) +
                        integrate_cf_dx(ng_setup['mesh'], ONE, ng_dx.pos_bulk())),
        },
        "area_interface_pos_gamma": {
            "description": "AREA (Interface) (1 dGamma_pos)", "type": "direct",
            "pc_form": integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], pc_dx.dGamma(side='+'), backend=backend),
            "ng_form": integrate_cf_dx(ng_setup['mesh'], ONE, ng_dx.dGamma()),
        },
        "area_interface_neg_gamma": {
            "description": "AREA (Interface) (1 dGamma_neg)", "type": "direct",
            "pc_form": integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], pc_dx.dGamma(side='-'), backend=backend),
            "ng_form": integrate_cf_dx(ng_setup['mesh'], ONE, ng_dx.dGamma()),
        },
        "area_interface_combined_gamma": {
            "description": "AREA (Interface) (dGamma_pos+dGamma_neg)", "type": "direct",
            "pc_form": (integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], pc_dx.dGamma(side='+'), backend=backend) +
                        integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], pc_dx.dGamma(side='-'), backend=backend)),
            "ng_form": (integrate_cf_dx(ng_setup['mesh'], ONE, ng_dx.dGamma()) +
                        integrate_cf_dx(ng_setup['mesh'], ONE, ng_dx.dGamma())),
        },
        "area_interface_gamma": {
            "description": "AREA (Interface) (1 dGamma)", "type": "direct",
            "pc_form": integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], pc_dx.dGamma(), backend=backend),
            "ng_form": integrate_cf_dx(ng_setup['mesh'], ONE, ng_dx.dGamma()),
        },
        "area_interface_pos_iface": {
            "description": "AREA (Interface) (1 pos_iface)", "type": "direct",
            "pc_form": integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], pc_dx.pos_iface(), backend=backend),
            "ng_form": integrate_cf_dx(ng_setup['mesh'], ONE, ng_dx.pos_iface()),
        },
        "area_interface_neg_iface": {
            "description": "AREA (Interface) (1 neg_iface)", "type": "direct",
            "pc_form": integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], pc_dx.neg_iface(), backend=backend),
            "ng_form": integrate_cf_dx(ng_setup['mesh'], ONE, ng_dx.neg_iface()),
        },
        "area_interface_combined_iface": {
            "description": "AREA (Interface) (diface_pos+diface_neg)", "type": "direct",
            "pc_form": (integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], pc_dx.pos_iface(), backend=backend) +
                        integrate_pc_constant_dx(pc_setup['dh'], pc_setup['ONE'], pc_dx.neg_iface(), backend=backend)),
            "ng_form": (integrate_cf_dx(ng_setup['mesh'], ONE, ng_dx.pos_iface()) +
                        integrate_cf_dx(ng_setup['mesh'], ONE, ng_dx.neg_iface())),
        },
        # --- Interface/Penalty Terms ---
        "nitsche": {
            "description": "NITSCHE (jump penalty)", "type": "total",
            "pc_form": (Constant(pc_setup['lam_val'])/Constant(maxh)) * pc_dot(pc_jump(pc_setup['up'],pc_setup['un']), pc_jump(pc_setup['vp'],pc_setup['vn'])) * pc_dx.dGamma(),
            "ng_form": (ng_setup['lam_val']/maxh) * (ng_setup['u'][0]-ng_setup['u'][1])*(ng_setup['v'][0]-ng_setup['v'][1]) * ng_dx.dGamma(),
        },
        
        # --- ================================== ---
        # --- Split-by-Region Versions of Terms ---
        # --- ================================== ---
        "mass_split": {
            "description": "MASS (u,v)", "type": "split", "parent": "mass_combined",
            "pc_forms": {
                ("pos","bulk"):    pc_inner(pc_setup['up'], pc_setup['vp']) * pc_dx.pos_bulk(),
                ("pos","interface"): pc_inner(pc_setup['up'], pc_setup['vp']) * pc_dx.pos_iface(),
                ("neg","bulk"):    pc_inner(pc_setup['un'], pc_setup['vn']) * pc_dx.neg_bulk(),
                ("neg","interface"): pc_inner(pc_setup['un'], pc_setup['vn']) * pc_dx.neg_iface(),
            },
            "ng_forms": {
                ("pos","bulk"):    InnerProduct(ng_setup['u'][1], ng_setup['v'][1]) * ng_dx.pos_bulk(),
                ("pos","interface"): InnerProduct(ng_setup['u'][1], ng_setup['v'][1]) * ng_dx.pos_iface(),
                ("neg","bulk"):    InnerProduct(ng_setup['u'][0], ng_setup['v'][0]) * ng_dx.neg_bulk(),
                ("neg","interface"): InnerProduct(ng_setup['u'][0], ng_setup['v'][0]) * ng_dx.neg_iface(),
            },
        },
        "volume_split": {
            "description": "VOLUME (2μ ε:ε)", "type": "split", "parent": "volume_combined",
            "pc_forms": {
                ("pos","bulk"):    2*pc_setup['mu1']*pc_inner(eps_pc(pc_setup['up']), eps_pc(pc_setup['vp']))*pc_dx.pos_bulk(),
                ("pos","interface"): 2*pc_setup['mu1']*pc_inner(eps_pc(pc_setup['up']), eps_pc(pc_setup['vp']))*pc_dx.pos_iface(),
                ("neg","bulk"):    2*pc_setup['mu0']*pc_inner(eps_pc(pc_setup['un']), eps_pc(pc_setup['vn']))*pc_dx.neg_bulk(),
                ("neg","interface"): 2*pc_setup['mu0']*pc_inner(eps_pc(pc_setup['un']), eps_pc(pc_setup['vn']))*pc_dx.neg_iface(),
            },
            "ng_forms": {
                ("pos","bulk"):    2*ng_setup['mu1']*InnerProduct(eps_ng(ng_setup['u'][1]), eps_ng(ng_setup['v'][1]))*ng_dx.pos_bulk(),
                ("pos","interface"): 2*ng_setup['mu1']*InnerProduct(eps_ng(ng_setup['u'][1]), eps_ng(ng_setup['v'][1]))*ng_dx.pos_iface(),
                ("neg","bulk"):    2*ng_setup['mu0']*InnerProduct(eps_ng(ng_setup['u'][0]), eps_ng(ng_setup['v'][0]))*ng_dx.neg_bulk(),
                ("neg","interface"): 2*ng_setup['mu0']*InnerProduct(eps_ng(ng_setup['u'][0]), eps_ng(ng_setup['v'][0]))*ng_dx.neg_iface(),
            },
        },
        "divu_q_split": {
            "description": "DIVU_Q", "type": "split", "parent": "divu_q",
            "pc_forms": {
                ("pos","bulk"):   (-pc_div(pc_setup['up'])*pc_setup['qp'])*pc_dx.pos_bulk(),
                ("pos","interface"): (-pc_div(pc_setup['up'])*pc_setup['qp'])*pc_dx.pos_iface(),
                ("neg","bulk"):   (-pc_div(pc_setup['un'])*pc_setup['qn'])*pc_dx.neg_bulk(),
                ("neg","interface"): (-pc_div(pc_setup['un'])*pc_setup['qn'])*pc_dx.neg_iface(),
            },
            "ng_forms": {
                ("pos","bulk"):   (-div(ng_setup['u'][1])*ng_setup['q'][1])*ng_dx.pos_bulk(),
                ("pos","interface"): (-div(ng_setup['u'][1])*ng_setup['q'][1])*ng_dx.pos_iface(),
                ("neg","bulk"):   (-div(ng_setup['u'][0])*ng_setup['q'][0])*ng_dx.neg_bulk(),
                ("neg","interface"): (-div(ng_setup['u'][0])*ng_setup['q'][0])*ng_dx.neg_iface(),
            },
        },
    }

    # --- 3. Execution Phase ---
    
    # ----- TOTAL term energies (FE vs FE) -----
    term_header("TOTAL term energies (PyCutFEM FE vs NGSolve FE)")
    results_total = {}
    for key, data in TEST_CASES.items():
        if not 'total' in data['type']  : continue
        
        E_pc = assemble_and_energy_pc(data['pc_form'], pc_setup['dh'], u_vec, v_vec, backend=backend)
        
        bf_ng = BilinearForm(ng_setup['WhG'], symmetric=False, check_unused=False)
        bf_ng += data['ng_form']
        E_ng = assemble_and_energy_ng(bf_ng, ng_setup['gfu'], ng_setup['gfv'])
        
        results_total[key] = {'pc': E_pc, 'ng': E_ng}
        err = srelerr(E_pc, E_ng)
        ok, mark = verdict(err, tol=1e-6)
        print(f"{data['description']:<35} | PC: {E_pc:+.12e}  NG: {E_ng:+.12e}  err = {err:.3e} {mark}")
        all_tests.append(TestResult(f"TOTAL {key}", err, ok, f"PC={E_pc:+.12e}, NG={E_ng:+.12e}"))
    # without bilinear forms
    for key, data in TEST_CASES.items():
        if not 'direct' in data['type']  : continue
        E_pc = data['pc_form']
        E_ng = data['ng_form']
        results_total[key] = {'pc': E_pc, 'ng': E_ng}
        err = srelerr(E_pc, E_ng)
        ok, mark = verdict(err, tol=1e-6)
        print(f"{data['description']:<35} | PC: {E_pc:+.12e}  NG: {E_ng:+.12e}  err = {err:.3e} {mark}")
        all_tests.append(TestResult(f"TOTAL {key}", err, ok, f"PC={E_pc:+.12e}, NG={E_ng:+.12e}"))

    # ----- SPLIT-by-region (PC FE vs NG FE) -----
    results_split = {}
    for key, data in TEST_CASES.items():
        if not 'split' in data['type']: continue
        term_header(f"{data['description']} (region split: NEG/POS × bulk/interface)")
        results_split[key] = {'pc': {}, 'ng': {}}
        for region_key in data['pc_forms'].keys():
            pc_form = data['pc_forms'][region_key]
            ng_form = data['ng_forms'][region_key]
            
            E_pc = assemble_and_energy_pc(pc_form, pc_setup['dh'], u_vec, v_vec, backend=backend)
            
            bf_ng = BilinearForm(ng_setup['WhG'], symmetric=False, check_unused=False)
            bf_ng += ng_form
            E_ng = assemble_and_energy_ng(bf_ng, ng_setup['gfu'], ng_setup['gfv'])
            
            results_split[key]['pc'][region_key] = E_pc
            results_split[key]['ng'][region_key] = E_ng

            err = srelerr(E_pc, E_ng)
            ok, mark = verdict(err, tol=1e-8)
            side, reg = region_key
            print(f"{side.upper():3s}/{reg:9s} : PC={E_pc:+.12e}   NG={E_ng:+.12e}   err={err:.3e} {mark}")
            all_tests.append(TestResult(f"SPLIT {key} {side}/{reg}", err, ok, f"PC={E_pc:+.12e}, NG={E_ng:+.12e}"))

    # ----- Consistency checks (PyCutFEM) -----
    term_header("Consistency checks (PyCutFEM)")
    for key, data in TEST_CASES.items():
        # Only run this check for items that are 'split' and have a 'parent'
        if 'split' in data.get('type') and 'parent' in data:
            parent_key = data['parent']
            
            # Ensure the parent key and the split key exist in the results
            if parent_key in results_total and key in results_split:
                total_from_split = sum(results_split[key]['pc'].values())
                total_from_direct = results_total[parent_key]['pc']
                
                err = srelerr(total_from_direct, total_from_split)
                ok, mark = verdict(err, tol=1e-12)
                print(f"total({parent_key}) vs Σsplit: {total_from_direct:+.12e} vs {total_from_split:+.12e} diff={err:.3e} {mark}")

    # ----- Summary -----
    failed = [t for t in all_tests if not t.passed]
    passed = [t for t in all_tests if t.passed]
    print(f"\n--- Test Summary ---\n{len(passed)} tests passed, {len(failed)} tests FAILED.\n")
    if failed:
        print("--- Top 6 Failed Tests (by error magnitude) ---")
        for t in sorted(failed, key=lambda z: z.error, reverse=True)[:6]:
            print(f"- {t.description}: error={t.error:.6e} | {t.details}")
    print("\nDone. Green ✓ indicates success within tolerance; red ✗ highlights discrepancies.")

if __name__ == "__main__":
    backend = os.getenv('PYCUTFEM_BACKEND', 'python').lower()  
    main(backend=backend)
