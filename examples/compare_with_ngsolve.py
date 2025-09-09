import numpy as np
import scipy.sparse.linalg as spla
from dataclasses import dataclass

# ... (keep all your existing imports) ...

# ---------- PyCutFEM imports ----------
from pycutfem.core.mesh import Mesh as pycutfem_Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    VectorTrialFunction, VectorTestFunction, TrialFunction, TestFunction,
    Function, VectorFunction, Constant,
    grad as pc_grad, inner as pc_inner, div as pc_div, dot as pc_dot, jump as pc_jump,
    Pos, Neg, CellDiameter,
)
from pycutfem.ufl.measures import dx as pycutfem_dx, dInterface as pycutfem_dInterface
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.core.levelset import CircleLevelSet
from pycutfem.fem.mixedelement import MixedElement

# ---------- NGSolve / XFEM imports ----------
from netgen.geom2d import SplineGeometry
from ngsolve import *
from xfem import *
from xfem.lsetcurv import *

# ---------- Data class to store test results ----------
@dataclass
class TestResult:
    description: str
    error: float
    passed: bool
    details: str

# ... (keep all your existing functions like srelerr, verdict, term_header, etc.) ...
def srelerr(a, b, rtol=1e-9, atol=1e-12):
    # Hybrid error: behaves like absolute error near zero and like relative error at scale.
    # err = |a-b| / max(atol, rtol*max(|a|, |b|))
    # return abs(a-b) / max(atol, rtol*max(abs(a), abs(b)))
    return abs(a-b)

def verdict(err, tol=1e-8):
    ok = err <= tol
    mark = "\x1b[32m✓\x1b[0m" if ok else "\x1b[31m✗\x1b[0m"
    return ok, mark

def term_header(title):
    print(f"\n=== {title} ===")

# ... (all your analytics, sympy, and build functions remain the same) ...
# ---------- Analytics (numpy and sympy) ----------
def make_analytics():
    """
    Return *numpy-callable* POS and NEG analytic fields so each side is different.
    Chosen as low-degree polynomials => exact symbolic integrals tractable.
    """
    import numpy as _np

    def u_pos(x, y):
        # vector field (degree 2)
        return _np.stack([x**2, x*y], axis=-1)
    def v_pos(x, y):
        return _np.stack([x, y**2], axis=-1)
    def p_pos(x, y):
        return x + y
    def q_pos(x, y):
        return x**2 - y

    def u_neg(x, y):
        # vector field (degree 2 with different pattern)
        return _np.stack([x*y, y**2], axis=-1)
    def v_neg(x, y):
        return _np.stack([x**2 - y, x - y**2], axis=-1)
    def p_neg(x, y):
        return x - y
    def q_neg(x, y):
        return x*y

    pos = dict(u=u_pos, v=v_pos, p=p_pos, q=q_pos)
    neg = dict(u=u_neg, v=v_neg, p=p_neg, q=q_neg)
    return pos, neg


def make_sympy_analytics():
    """
    Return *sympy* expressions (not callables) for the same POS/NEG fields.
    """
    import sympy as sp
    x, y = sp.symbols('x y', real=True)

    u_pos = (x**2, x*y)
    v_pos = (x, y**2)
    p_pos = x + y
    q_pos = x**2 - y

    u_neg = (x*y, y**2)
    v_neg = (x**2 - y, x - y**2)
    p_neg = x - y
    q_neg = x*y

    return (x, y), dict(u=u_pos, v=v_pos, p=p_pos, q=q_pos), dict(u=u_neg, v=v_neg, p=p_neg, q=q_neg)

# ---------- Sympy exact integrals for reference ----------
def sympy_eps(u):
    import sympy as sp
    x, y = sp.symbols('x y', real=True)
    ux, uy = u
    dux = (sp.diff(ux, x), sp.diff(ux, y))
    duy = (sp.diff(uy, x), sp.diff(uy, y))
    # grad u = [[ux_x, ux_y],[uy_x, uy_y]]
    G = ((dux[0], dux[1]), (duy[0], duy[1]))
    # symmetrized
    E = ((G[0][0], (G[0][1]+G[1][0])/2),
         ((G[1][0]+G[0][1])/2, G[1][1]))
    return E

def sympy_inner_mat(A, B):
    return sum(A[i][j]*B[i][j] for i in range(2) for j in range(2))

def sympy_div(u):
    import sympy as sp
    x, y = sp.symbols('x y', real=True)
    ux, uy = u
    return sp.diff(ux, x) + sp.diff(uy, y)

def _integrate_square(expr, x, y):
    import sympy as sp
    return sp.integrate(sp.integrate(expr, (y, -1, 1)), (x, -1, 1))

def _integrate_disk(expr, x, y, R):
    import sympy as sp
    r, th = sp.symbols('r th', nonnegative=True, real=True)
    expr_polar = sp.simplify(expr.subs({x: r*sp.cos(th), y: r*sp.sin(th)}))
    return sp.integrate(sp.integrate(expr_polar * r, (r, 0, R)), (th, 0, 2*sp.pi))

def sympy_exact_energies(R, mu0=1.0, mu1=10.0):
    """
    Return exact integrals for:
      - VOLUME: 2*mu*<eps(u), eps(v)> over NEG (disk) and POS (square\disk)
      - DIVU_Q: (-div(u))*q
      - DIVV_P: (-div(v))*p
    Using the polynomial analytics from make_sympy_analytics().
    """
    import sympy as sp
    (x, y), pos, neg = make_sympy_analytics()

    # NEG (disk)
    Eneg_u = sympy_eps(neg['u']); Eneg_v = sympy_eps(neg['v'])
    I_vol_neg = 2*mu0*sympy_inner_mat(Eneg_u, Eneg_v)
    I_divu_q_neg = - sympy_div(neg['u']) * neg['q']
    I_divv_p_neg = - sympy_div(neg['v']) * neg['p']

    Vol_neg = sp.simplify(_integrate_disk(I_vol_neg, x, y, R))
    DivuQ_neg = sp.simplify(_integrate_disk(I_divu_q_neg, x, y, R))
    DivvP_neg = sp.simplify(_integrate_disk(I_divv_p_neg, x, y, R))

    # POS (square minus disk)
    Epos_u = sympy_eps(pos['u']); Epos_v = sympy_eps(pos['v'])
    I_vol_pos = 2*mu1*sympy_inner_mat(Epos_u, Epos_v)
    I_divu_q_pos = - sympy_div(pos['u']) * pos['q']
    I_divv_p_pos = - sympy_div(pos['v']) * pos['p']

    Vol_pos_square = sp.simplify(_integrate_square(I_vol_pos, x, y))
    DivuQ_pos_square = sp.simplify(_integrate_square(I_divu_q_pos, x, y))
    DivvP_pos_square = sp.simplify(_integrate_square(I_divv_p_pos, x, y))

    Vol_pos_disk = sp.simplify(_integrate_disk(I_vol_pos, x, y, R))
    DivuQ_pos_disk = sp.simplify(_integrate_disk(I_divu_q_pos, x, y, R))
    DivvP_pos_disk = sp.simplify(_integrate_disk(I_divv_p_pos, x, y, R))

    Vol_pos = sp.simplify(Vol_pos_square - Vol_pos_disk)
    DivuQ_pos = sp.simplify(DivuQ_pos_square - DivuQ_pos_disk)
    DivvP_pos = sp.simplify(DivvP_pos_square - DivvP_pos_disk)

    # totals
    Vol_tot = sp.simplify(Vol_neg + Vol_pos)
    DivuQ_tot = sp.simplify(DivuQ_neg + DivuQ_pos)
    DivvP_tot = sp.simplify(DivvP_neg + DivvP_pos)

    return dict(
        volume = dict(neg=Vol_neg, pos=Vol_pos, total=Vol_tot),
        divu_q = dict(neg=DivuQ_neg, pos=DivuQ_pos, total=DivuQ_tot),
        divv_p = dict(neg=DivvP_neg, pos=DivvP_pos, total=DivvP_tot),
    )

# ---------- PyCutFEM build & split-forms ----------
def build_pc(maxh=0.125, order=2, R=2.0/3.0):
    L = H = 2.0
    nx = int(L / maxh)
    nodes, elems, _, corners = structured_quad(
        L, H, nx=nx, ny=nx, poly_order=1, offset=[-L/2, -H/2]
    )
    mesh = pycutfem_Mesh(nodes, elems, elements_corner_nodes=corners,
                         element_type="quad", poly_order=1)
    level_set = CircleLevelSet(center=(0.0, 0.0), radius=R)

    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)

    inside   = mesh.element_bitset("inside")    # NEG
    outside  = mesh.element_bitset("outside")   # POS
    interface= mesh.element_bitset("cut")

    has_inside  = inside  | interface
    has_outside = outside | interface

    me = MixedElement(mesh, field_specs={
        'u_pos_x': order, 'u_pos_y': order, 'p_pos_': order-1,
        'u_neg_x': order, 'u_neg_y': order, 'p_neg_': order-1,
        'lm': ':number:'
    })
    dh = DofHandler(me, method='cg')

    Vpos = FunctionSpace("Vpos", ['u_pos_x','u_pos_y'], side='+')
    Vneg = FunctionSpace("Vneg", ['u_neg_x','u_neg_y'], side='-')
    up, vp = VectorTrialFunction(Vpos, dh, side='+'), VectorTestFunction(Vpos, dh, side='+')
    un, vn = VectorTrialFunction(Vneg, dh, side='-'), VectorTestFunction(Vneg, dh, side='-')
    pp, qp = TrialFunction('p_pos_', dh, side='+'), TestFunction('p_pos_', dh, side='+')
    pn, qn = TrialFunction('p_neg_', dh, side='-'), TestFunction('p_neg_', dh, side='-')
    nL, mL = TrialFunction('lm', dh), TestFunction('lm', dh)

    qvol = 2*order + 4
    dx_pos_all = pycutfem_dx(defined_on=has_outside, level_set=level_set, metadata={'side': '+', 'q': qvol})
    dx_neg_all = pycutfem_dx(defined_on=has_inside,  level_set=level_set, metadata={'side': '-', 'q': qvol})

    # split measures (bulk vs interface)
    dx_pos_bulk = pycutfem_dx(defined_on=outside,   level_set=level_set, metadata={'side': '+', 'q': qvol})
    dx_pos_iface= pycutfem_dx(defined_on=interface, level_set=level_set, metadata={'side': '+', 'q': qvol})
    dx_neg_bulk = pycutfem_dx(defined_on=inside,    level_set=level_set, metadata={'side': '-', 'q': qvol})
    dx_neg_iface= pycutfem_dx(defined_on=interface, level_set=level_set, metadata={'side': '-', 'q': qvol})

    dGamma = pycutfem_dInterface(defined_on=interface, level_set=level_set, metadata={'q': qvol})

    mu0, mu1 = Constant(1.0), Constant(10.0)
    lam = Constant(0.5*(mu0.value+mu1.value)*20*order**2)
    h = CellDiameter()

    def eps(u): return 0.5*(pc_grad(u)+pc_grad(u).T)

    def K_of(form):
        K,_ = assemble_form(Equation(form, None), dof_handler=dh, quad_order=qvol, backend='python')
        return K

    # total terms
    A_vol = (2*mu1*pc_inner(eps(up), eps(vp))*dx_pos_all
           + 2*mu0*pc_inner(eps(un), eps(vn))*dx_neg_all)

    # split "mixed" into two separate terms
    A_divu_q = (- pc_div(up)*qp) * dx_pos_all + (- pc_div(un)*qn) * dx_neg_all
    A_divv_p = (- pc_div(vp)*pp) * dx_pos_all + (- pc_div(vn)*pn) * dx_neg_all

    # keep the original total mixed term for compatibility
    A_mix = A_divu_q + A_divv_p

    A_mean = (nL*Neg(qn) + mL*Neg(pn))*dx_neg_all
    A_nitsche = (lam/Constant(0.125)) * pc_dot(pc_jump(up,un), pc_jump(vp,vn)) * dGamma

    terms_pc = {
        "volume":  K_of(A_vol),
        "mixed":   K_of(A_mix),
        "divu_q":  K_of(A_divu_q),
        "divv_p":  K_of(A_divv_p),
        "mean":    K_of(A_mean),
        "nitsche": K_of(A_nitsche),
    }

    # split terms (bulk/interface per side)
    A_vol_split = {
        ("pos","bulk"):   2*mu1*pc_inner(eps(up), eps(vp))*dx_pos_bulk,
        ("pos","interface"): 2*mu1*pc_inner(eps(up), eps(vp))*dx_pos_iface,
        ("neg","bulk"):   2*mu0*pc_inner(eps(un), eps(vn))*dx_neg_bulk,
        ("neg","interface"): 2*mu0*pc_inner(eps(un), eps(vn))*dx_neg_iface,
    }
    A_divu_q_split = {
        ("pos","bulk"):   (- pc_div(up)*qp)*dx_pos_bulk,
        ("pos","interface"): (- pc_div(up)*qp)*dx_pos_iface,
        ("neg","bulk"):   (- pc_div(un)*qn)*dx_neg_bulk,
        ("neg","interface"): (- pc_div(un)*qn)*dx_neg_iface,
    }
    A_divv_p_split = {
        ("pos","bulk"):   (- pc_div(vp)*pp)*dx_pos_bulk,
        ("pos","interface"): (- pc_div(vp)*pp)*dx_pos_iface,
        ("neg","bulk"):   (- pc_div(vn)*pn)*dx_neg_bulk,
        ("neg","interface"): (- pc_div(vn)*pn)*dx_neg_iface,
    }
    A_mix_split = {k: A_divu_q_split[k] + A_divv_p_split[k] for k in A_divu_q_split}

    A_mean_split = {
        ("neg","bulk"):   (nL*Neg(qn) + mL*Neg(pn))*dx_neg_bulk,
        ("neg","interface"): (nL*Neg(qn) + mL*Neg(pn))*dx_neg_iface,
    }

    terms_pc_split = {
        "volume": {k: K_of(v) for k,v in A_vol_split.items()},
        "divu_q": {k: K_of(v) for k,v in A_divu_q_split.items()},
        "divv_p": {k: K_of(v) for k,v in A_divv_p_split.items()},
        "mixed":  {k: K_of(v) for k,v in A_mix_split.items()},
        "mean":   {k: K_of(v) for k,v in A_mean_split.items()},
    }

    # area(1) checks: assemble bilinear form of 1*dx on NumberSpace
    ONE = Constant(1.0)
    A_one_split = {
        ("pos","bulk"):    K_of(ONE*dx_pos_bulk),
        ("pos","interface"):K_of(ONE*dx_pos_iface),
        ("neg","bulk"):    K_of(ONE*dx_neg_bulk),
        ("neg","interface"):K_of(ONE*dx_neg_iface),
    }

    # pack vectors placeholders (filled later)
    u_pos = VectorFunction("u_pos", ['u_pos_x','u_pos_y'], dh, side='+')
    v_pos = VectorFunction("v_pos", ['u_pos_x','u_pos_y'], dh, side='+')
    u_neg = VectorFunction("u_neg", ['u_neg_x','u_neg_y'], dh, side='-')
    v_neg = VectorFunction("v_neg", ['u_neg_x','u_neg_y'], dh, side='-')
    p_pos = Function("p_pos", 'p_pos_', dh, side='+')
    q_pos = Function("q_pos", 'p_pos_', dh, side='+')
    p_neg = Function("p_neg", 'p_neg_', dh, side='-')
    q_neg = Function("q_neg", 'p_neg_', dh, side='-')

    return dict(mesh=mesh, dh=dh,
                terms=terms_pc,
                terms_split=terms_pc_split,
                area_split=A_one_split,
                fields=dict(u_pos=u_pos, v_pos=v_pos, u_neg=u_neg, v_neg=v_neg,
                            p_pos=p_pos, q_pos=q_pos, p_neg=p_neg, q_neg=q_neg))

def build_pc_whole_domain(maxh=0.125, order=2):
    """
    Build a PyCutFEM problem *without* any level-set usage for whole-domain tests.
    We keep the same MixedElement (POS/NEG fields exist), but integrate only
    the POS fields over plain dx (no level_set) so the path is strictly 'no interface'.
    """
    L = H = 2.0
    nx = int(L / maxh)
    nodes, elems, _, corners = structured_quad(
        L, H, nx=nx, ny=nx, poly_order=1, offset=[-L/2, -H/2]
    )
    mesh = pycutfem_Mesh(nodes, elems, elements_corner_nodes=corners,
                         element_type="quad", poly_order=1)

    me = MixedElement(mesh, field_specs={
        'u_pos_x': order, 'u_pos_y': order, 'p_pos_': order-1,
        'u_neg_x': order, 'u_neg_y': order, 'p_neg_': order-1,
        'lm': ':number:'
    })
    dh = DofHandler(me, method='cg')

    Vpos = FunctionSpace("Vpos", ['u_pos_x','u_pos_y'], side='+')
    up, vp = VectorTrialFunction(Vpos, dh, side='+'), VectorTestFunction(Vpos, dh, side='+')
    pp, qp = TrialFunction('p_pos_', dh, side='+'), TestFunction('p_pos_', dh, side='+')

    qvol = 2*order + 4
    dx_all = pycutfem_dx(metadata={'q': qvol})  # no defined_on, no level_set

    mu1 = Constant(10.0)

    def eps(u): return 0.5*(pc_grad(u)+pc_grad(u).T)

    def K_of(form):
        K,_ = assemble_form(Equation(form, None), dof_handler=dh, quad_order=qvol, backend='python')
        return K

    A_vol_pos_whole  = 2*mu1*pc_inner(eps(up), eps(vp))*dx_all
    A_divu_q_whole   = (- pc_div(up)*qp) * dx_all
    A_divv_p_whole   = (- pc_div(vp)*pp) * dx_all

    terms = {
        "volume_pos_whole": K_of(A_vol_pos_whole),
        "divu_q_whole":     K_of(A_divu_q_whole),
        "divv_p_whole":     K_of(A_divv_p_whole),
    }

    # pack vectors as usual
    u_pos = VectorFunction("u_pos", ['u_pos_x','u_pos_y'], dh, side='+')
    v_pos = VectorFunction("v_pos", ['u_pos_x','u_pos_y'], dh, side='+')
    p_pos = Function("p_pos", 'p_pos_', dh, side='+')
    q_pos = Function("q_pos", 'p_pos_', dh, side='+')
    u_neg = VectorFunction("u_neg", ['u_neg_x','u_neg_y'], dh, side='-')
    v_neg = VectorFunction("v_neg", ['u_neg_x','u_neg_y'], dh, side='-')
    p_neg = Function("p_neg", 'p_neg_', dh, side='-')
    q_neg = Function("q_neg", 'p_neg', dh, side='-') if False else None  # placeholder, not used

    return dict(mesh=mesh, dh=dh, terms=terms,
                fields=dict(u_pos=u_pos, v_pos=v_pos, p_pos=p_pos, q_pos=q_pos,
                            u_neg=u_neg, v_neg=v_neg, p_neg=p_neg))

def inject_pc(d, pos, neg):
    """
    Inject separate numpy-callable POS/NEG fields into the PyCutFEM Function containers.
    pos, neg = dict(u=vec_fun, v=vec_fun, p=scal_fun, q=scal_fun)
    """
    # Vector fields
    d['fields']['u_pos'].set_values_from_function(pos['u'])
    d['fields']['v_pos'].set_values_from_function(pos['v'])
    d['fields']['u_neg'].set_values_from_function(neg['u'])
    d['fields']['v_neg'].set_values_from_function(neg['v'])
    # Scalar fields
    d['fields']['p_pos'].set_values_from_function(pos['p'])
    d['fields']['q_pos'].set_values_from_function(pos['q'])
    d['fields']['p_neg'].set_values_from_function(neg['p'])
    if d['fields'].get('q_neg', None) is not None:
        d['fields']['q_neg'].set_values_from_function(neg['q'])

    # build global vectors (ordering = MixedElement)
    dh = d['dh']; total = dh.total_dofs
    def pack(Upos, Ppos, Uneg, Pneg, lm_val=0.0):
        vec = np.zeros(total)
        sl = dh.get_field_slice

        vec[sl('u_pos_x')] = Upos.components[0].nodal_values
        vec[sl('u_pos_y')] = Upos.components[1].nodal_values

        vec[sl('u_neg_x')] = Uneg.components[0].nodal_values
        vec[sl('u_neg_y')] = Uneg.components[1].nodal_values

        vec[sl('p_pos_')] = Ppos.nodal_values
        vec[sl('p_neg_')] = Pneg.nodal_values

        vec[sl('lm')] = lm_val
        return vec

    u_vec = pack(d['fields']['u_pos'], d['fields']['p_pos'],
                 d['fields']['u_neg'], d['fields']['p_neg'], lm_val=0.7)
    # v_vec uses q_* in pressure slots by design
    v_vec = pack(d['fields']['v_pos'], d['fields']['q_pos'],
                 d['fields']['v_neg'], d['fields'].get('q_neg', d['fields']['p_neg']), lm_val=-0.3)

    return u_vec, v_vec

# ---------- NGSolve helpers ----------
def ng_cf_grad_vec(U):
    return CoefficientFunction((
        U[0].Diff(x), U[0].Diff(y),
        U[1].Diff(x), U[1].Diff(y)
    ), dims=(2,2))

def ng_cf_eps(U):
    G = ng_cf_grad_vec(U)
    return 0.5*(G + G.trans)

def ng_cf_div(U):
    return U[0].Diff(x) + U[1].Diff(y)

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

def build_ng(maxh=0.125, order=2, R=2.0/3.0, no_dirichlet=True):
    square = SplineGeometry()
    square.AddRectangle((-1, -1), (1, 1), bcs=[1, 2, 3, 4])
    mesh = Mesh(square.GenerateMesh(maxh=maxh, quad_dominated=True))

    # level-set
    rsqr = x**2 + y**2
    r = sqrt(rsqr)
    levelset = r - R
    lsetp1 = GridFunction(H1(mesh,order=1))
    InterpolateToP1(levelset, lsetp1)
    ci = CutInfo(mesh, lsetp1)

    Vhbase = VectorH1(mesh, order=order) if no_dirichlet else VectorH1(mesh, order=order, dirichlet=[1,2,3,4])
    Qhbase = H1(mesh, order=order-1)
    Vhneg = Compress(Vhbase, GetDofsOfElements(Vhbase, ci.GetElementsOfType(HASNEG)))
    Vhpos = Compress(Vhbase, GetDofsOfElements(Vhbase, ci.GetElementsOfType(HASPOS)))
    Qhneg = Compress(Qhbase, GetDofsOfElements(Qhbase, ci.GetElementsOfType(HASNEG)))
    Qhpos = Compress(Qhbase, GetDofsOfElements(Qhbase, ci.GetElementsOfType(HASPOS)))
    WhG = FESpace([Vhneg*Vhpos, Qhneg*Qhpos, NumberSpace(mesh)], dgjumps=True)

    dx_neg_all = dCut(lsetp1, NEG)
    dx_pos_all = dCut(lsetp1, POS)
    ds         = dCut(lsetp1, IF)

    mu0, mu1 = 1.0, 10.0
    lam = 0.5*(mu0+mu1)*20*order*order

    # FE bilinear forms (v^T K u) on NG
    u, p, n = WhG.TrialFunction()
    v, q, m = WhG.TestFunction()
    def eps(u): return 0.5*(Grad(u)+Grad(u).trans)

    a_vol = BilinearForm(WhG, symmetric=False)
    a_vol += 2*mu0*InnerProduct(eps(u[0]), eps(v[0]))*dx_neg_all
    a_vol += 2*mu1*InnerProduct(eps(u[1]), eps(v[1]))*dx_pos_all

    # split mixed terms
    a_divu_q = BilinearForm(WhG, symmetric=False)
    a_divu_q += (- div(u[0])*q[0]) * dx_neg_all
    a_divu_q += (- div(u[1])*q[1]) * dx_pos_all

    a_divv_p = BilinearForm(WhG, symmetric=False)
    a_divv_p += (- div(v[0])*p[0]) * dx_neg_all
    a_divv_p += (- div(v[1])*p[1]) * dx_pos_all

    a_mix = BilinearForm(WhG, symmetric=False)
    a_mix += (- div(u[0])*q[0] - div(v[0])*p[0]) * dx_neg_all
    a_mix += (- div(u[1])*q[1] - div(v[1])*p[1]) * dx_pos_all

    a_mean = BilinearForm(WhG, symmetric=False)
    a_mean += (n*q[0] + m*p[0]) * dx_neg_all

    a_nitsche = BilinearForm(WhG, symmetric=False)
    a_nitsche += (lam/0.125) * (u[0]-u[1])*(v[0]-v[1]) * ds

    # exact CFs (symbolic)
    # Using same polynomials as numpy analytics
    U_pos = CoefficientFunction(( x**2, x*y ))
    V_pos = CoefficientFunction(( x,   y**2 ))
    P_pos = CoefficientFunction( x + y )
    Q_pos = CoefficientFunction( x**2 - y )

    U_neg = CoefficientFunction(( x*y, y**2 ))
    V_neg = CoefficientFunction(( x**2 - y, x - y**2 ))
    P_neg = CoefficientFunction( x - y )
    Q_neg = CoefficientFunction( x*y )

    # FE gridfunctions for energies
    gfu = GridFunction(WhG)
    gfv = GridFunction(WhG)
    vel_u = gfu.components[0]; pre_u = gfu.components[1]; lm_u = gfu.components[2]
    vel_v = gfv.components[0]; pre_v = gfv.components[1]; lm_v = gfv.components[2]
    # assign pos/neg components separately
    vel_u.components[0].Set(U_neg)  # NEG side
    vel_u.components[1].Set(U_pos)  # POS side
    vel_v.components[0].Set(V_neg)
    vel_v.components[1].Set(V_pos)
    pre_u.components[0].Set(P_neg)
    pre_u.components[1].Set(P_pos)
    pre_v.components[0].Set(Q_neg)
    pre_v.components[1].Set(Q_pos)
    lm_u.Set(CoefficientFunction(0.7))
    lm_v.Set(CoefficientFunction(-0.3))

    # region splits
    dx_neg_bulk   = dCut(lsetp1, NEG, definedonelements=ci.GetElementsOfType(NEG))
    dx_pos_bulk   = dCut(lsetp1, POS, definedonelements=ci.GetElementsOfType(POS))
    dx_neg_iface  = dCut(lsetp1, NEG, definedonelements=ci.GetElementsOfType(IF))
    dx_pos_iface  = dCut(lsetp1, POS, definedonelements=ci.GetElementsOfType(IF))

    def ng_split_energies_cf():
        # volume split
        E_vol = {
          ("neg","bulk"):   integrate_cf_dx(mesh, 2*mu0*InnerProduct(ng_cf_eps(U_neg), ng_cf_eps(V_neg)), dx_neg_bulk),
          ("neg","interface"): integrate_cf_dx(mesh, 2*mu0*InnerProduct(ng_cf_eps(U_neg), ng_cf_eps(V_neg)), dx_neg_iface),
          ("pos","bulk"):   integrate_cf_dx(mesh, 2*mu1*InnerProduct(ng_cf_eps(U_pos), ng_cf_eps(V_pos)), dx_pos_bulk),
          ("pos","interface"): integrate_cf_dx(mesh, 2*mu1*InnerProduct(ng_cf_eps(U_pos), ng_cf_eps(V_pos)), dx_pos_iface),
        }
        # mixed split – individual pieces
        E_divu_q = {
          ("neg","bulk"):   integrate_cf_dx(mesh, (-ng_cf_div(U_neg))*Q_neg, dx_neg_bulk),
          ("neg","interface"): integrate_cf_dx(mesh, (-ng_cf_div(U_neg))*Q_neg, dx_neg_iface),
          ("pos","bulk"):   integrate_cf_dx(mesh, (-ng_cf_div(U_pos))*Q_pos, dx_pos_bulk),
          ("pos","interface"): integrate_cf_dx(mesh, (-ng_cf_div(U_pos))*Q_pos, dx_pos_iface),
        }
        E_divv_p = {
          ("neg","bulk"):   integrate_cf_dx(mesh, (-ng_cf_div(V_neg))*P_neg, dx_neg_bulk),
          ("neg","interface"): integrate_cf_dx(mesh, (-ng_cf_div(V_neg))*P_neg, dx_neg_iface),
          ("pos","bulk"):   integrate_cf_dx(mesh, (-ng_cf_div(V_pos))*P_pos, dx_pos_bulk),
          ("pos","interface"): integrate_cf_dx(mesh, (-ng_cf_div(V_pos))*P_pos, dx_pos_iface),
        }
        E_mix = {k: E_divu_q[k]+E_divv_p[k] for k in E_divu_q}

        # area(1) checks
        ONE = CoefficientFunction(1.0)
        A_one = {
          ("neg","bulk"):   integrate_cf_dx(mesh, ONE, dx_neg_bulk),
          ("neg","interface"): integrate_cf_dx(mesh, ONE, dx_neg_iface),
          ("pos","bulk"):   integrate_cf_dx(mesh, ONE, dx_pos_bulk),
          ("pos","interface"): integrate_cf_dx(mesh, ONE, dx_pos_iface),
        }
        return E_vol, E_divu_q, E_divv_p, E_mix, A_one

    # --- whole-domain (no interface) measures ---
    dx_all = dx  # standard domain measure (no dCut)
    def ng_whole_cf():
        E_vol_pos_whole = integrate_cf_dx(mesh, 2*mu1*InnerProduct(ng_cf_eps(U_pos), ng_cf_eps(V_pos)), dx_all)
        E_divu_q_whole  = integrate_cf_dx(mesh, (-ng_cf_div(U_pos))*Q_pos, dx_all)
        E_divv_p_whole  = integrate_cf_dx(mesh, (-ng_cf_div(V_pos))*P_pos, dx_all)
        return dict(volume_pos_whole=E_vol_pos_whole,
                    divu_q_whole=E_divu_q_whole,
                    divv_p_whole=E_divv_p_whole)

    terms_ng = {
        "volume": a_vol, "mixed": a_mix,
        "divu_q": a_divu_q, "divv_p": a_divv_p,
        "mean": a_mean, "nitsche": a_nitsche
    }
    return dict(mesh=mesh, WhG=WhG, terms=terms_ng, gfu=gfu, gfv=gfv,
                split_cf=ng_split_energies_cf,
                whole_cf=ng_whole_cf)

def assemble_and_energy_ng(bf: BilinearForm, gfu, gfv):
    with TaskManager():
        bf.Assemble()
    tmp = gfv.vec.CreateVector()
    bf.Apply(gfu.vec, tmp)
    return InnerProduct(gfv.vec, tmp)

def assemble_and_energy_pc(K, u_vec, v_vec):
    return float(v_vec @ (K @ u_vec))

# ---------- Runner ----------
def main():
    maxh, order, R = 0.125, 2, 2.0/3.0
    R_big = False
    if R_big:
        R = 10.0

    try:
        print("importing ngsxfem-", GetXDGVersion())
    except Exception:
        pass
    
    test_results = []
    pos_np, neg_np = make_analytics()
    pc = build_pc(maxh, order, R)
    ng = build_ng(maxh, order, R, no_dirichlet=True)
    u_vec, v_vec = inject_pc(pc, pos_np, neg_np)

    term_map = {
        "volume": "VOLUME (2 μ ⟨ε(u), ε(v)⟩)", "divu_q": "DIVU_Q (- div(u) · q)",
        "divv_p": "DIVV_P (- div(v) · p)", "mixed": "MIXED (sum of DIVU_Q + DIVV_P)",
        "mean": "MEAN (neg-side averages)", "nitsche": "NITSCHE (jump penalty on interface)",
    }

    term_header("TOTAL term energies (PyCutFEM FE vs NGSolve FE)")
    for key in ("volume", "divu_q", "divv_p", "mixed", "mean", "nitsche"):
        E_pc = assemble_and_energy_pc(pc["terms"][key], u_vec, v_vec)
        E_ng = assemble_and_energy_ng(ng["terms"][key], ng["gfu"], ng["gfv"])
        err = srelerr(E_pc, E_ng)
        ok, mark = verdict(err, tol=1e-9)
        details = f"{term_map[key]:28s} | PC: {E_pc:+.12e}  NG: {E_ng:+.12e}  scaled.err = {err:.3e} {mark}"
        print(details)
        test_results.append(TestResult(f"TOTAL: {term_map[key]}", err, ok, details))

    E_vol_cf, E_divu_q_cf, E_divv_p_cf, E_mix_cf, _ = ng["split_cf"]()
    E_vol_pc = {k: assemble_and_energy_pc(v, u_vec, v_vec) for k, v in pc["terms_split"]["volume"].items()}
    E_divu_q_pc = {k: assemble_and_energy_pc(v, u_vec, v_vec) for k, v in pc["terms_split"]["divu_q"].items()}
    E_divv_p_pc = {k: assemble_and_energy_pc(v, u_vec, v_vec) for k, v in pc["terms_split"]["divv_p"].items()}
    E_mix_pc = {k: assemble_and_energy_pc(v, u_vec, v_vec) for k, v in pc["terms_split"]["mixed"].items()}

    def prsplit(title, pcmap, ngmap):
        term_header(title + " (region split: NEG/POS × bulk/interface)")
        for side in ("neg", "pos"):
            for reg in ("bulk", "interface"):
                key = (side, reg)
                if key in pcmap and key in ngmap:
                    a, b = pcmap[key], ngmap[key]
                    err = srelerr(a, b)
                    ok, mark = verdict(err, tol=1e-8)
                    details = f"{side.upper():3s}/{reg:9s} : PC={a:+.12e}   NG(dx)={b:+.12e}   scaled.err={err:.3e} {mark}"
                    print(details)
                    test_results.append(TestResult(f"{title} SPLIT: {side.upper()}/{reg}", err, ok, details))

    prsplit("VOLUME", E_vol_pc, E_vol_cf)
    prsplit("DIVU_Q", E_divu_q_pc, E_divu_q_cf)
    prsplit("DIVV_P", E_divv_p_pc, E_divv_p_cf)
    prsplit("MIXED", E_mix_pc, E_mix_cf)

    term_header("Consistency checks")
    tot_pc = {name: assemble_and_energy_pc(pc["terms"][name], u_vec, v_vec) for name in ("volume", "divu_q", "divv_p", "mixed")}
    sum_splits = {"volume": sum(E_vol_pc.values()), "divu_q": sum(E_divu_q_pc.values()), "divv_p": sum(E_divv_p_pc.values()), "mixed": sum(E_mix_pc.values())}
    
    for k in tot_pc:
        err = srelerr(tot_pc[k], sum_splits[k])
        ok, mark = verdict(err, tol=1e-12)
        details = f"total({k}) vs Σsplit: {tot_pc[k]:+.12e}  vs  {sum_splits[k]:+.12e}  scaled.diff={err:.3e} {mark}"
        print(details)
        test_results.append(TestResult(f"Consistency: total({k}) vs Σsplit", err, ok, details))

    err = srelerr(tot_pc['mixed'], tot_pc['divu_q'] + tot_pc['divv_p'])
    ok, mark = verdict(err, tol=1e-14)
    details = f"mixed vs (divu_q + divv_p): {tot_pc['mixed']:+.12e}  vs  {(tot_pc['divu_q']+tot_pc['divv_p']):+.12e}  scaled.diff={err:.3e} {mark}"
    print(details)
    test_results.append(TestResult("Consistency: mixed vs (divu_q + divv_p)", err, ok, details))

    Evu = tot_pc["volume"]
    Evu_swapped = float(u_vec @ (pc["terms"]["volume"] @ v_vec))
    err = srelerr(Evu, Evu_swapped)
    ok, mark = verdict(err, tol=1e-14)
    details = f"Symmetry check (volume): E(u,v)={Evu:+.12e} vs E(v,u)={Evu_swapped:+.12e}  scaled.diff={err:.3e} {mark}"
    print(details)
    test_results.append(TestResult("Consistency: Symmetry check (volume)", err, ok, details))

    term_header("Exact comparisons (sympy) for VOLUME, DIVU_Q, DIVV_P over NEG (disk), POS (square\\disk), TOTAL")
    exact = sympy_exact_energies(R)
    for key in ("volume", "divu_q", "divv_p"):
        print(f"Evaluating exact {term_map[key]} ...")
        Epc = assemble_and_energy_pc(pc["terms"][key], u_vec, v_vec)
        import sympy as sp
        En_pos, En_neg, En_tot = float(sp.N(exact[key]['pos'])), float(sp.N(exact[key]['neg'])), float(sp.N(exact[key]['total']))
        err_tot = srelerr(Epc, En_tot)
        ok, mark = verdict(err_tot, tol=1e-8)
        details = f"{term_map[key]:28s} | PC(total)={Epc:+.12e}  EX(total)={En_tot:+.12e}  scaled.err={err_tot:.3e} {mark}"
        print(details)
        test_results.append(TestResult(f"EXACT: {term_map[key]} TOTAL", err_tot, ok, details))

        pc_side = E_vol_pc if key == "volume" else (E_divu_q_pc if key == "divu_q" else E_divv_p_pc)
        a_neg = sum(val for (s, reg), val in pc_side.items() if s == "neg")
        a_pos = sum(val for (s, reg), val in pc_side.items() if s == "pos")
        
        err_neg = srelerr(a_neg, En_neg)
        okn, markn = verdict(err_neg, tol=1e-8)
        details_neg = f"    NEG side: PC={a_neg:+.12e}  EX={En_neg:+.12e}  scaled.err={err_neg:.3e} {markn}"
        print(details_neg)
        test_results.append(TestResult(f"EXACT: {key} NEG side", err_neg, okn, details_neg))

        err_pos = srelerr(a_pos, En_pos)
        okp, markp = verdict(err_pos, tol=1e-8)
        details_pos = f"    POS side: PC={a_pos:+.12e}  EX={En_pos:+.12e}  scaled.err={err_pos:.3e} {markp}"
        print(details_pos)
        test_results.append(TestResult(f"EXACT: {key} POS side", err_pos, okp, details_pos))

    term_header("Whole-domain (no interface) — PyCutFEM(dx) vs NGSolve(dx) vs Exact(dx)")
    pc_whole = build_pc_whole_domain(maxh, order)
    pos_np, _ = make_analytics()
    u_vec_w, v_vec_w = inject_pc(pc_whole, pos_np, {'u': pos_np['u'], 'v': pos_np['v'], 'p': pos_np['p'], 'q': pos_np['q']})
    ng_whole = ng["whole_cf"]()

    from sympy import simplify
    (x, y), pos_sym, _ = make_sympy_analytics()
    def exact_whole_from_sympy():
        import sympy as sp
        _int_square = lambda e: sp.integrate(sp.integrate(e, (y, -1, 1)), (x, -1, 1))
        Epos_u, Epos_v = sympy_eps(pos_sym['u']), sympy_eps(pos_sym['v'])
        I_vol = 2 * 10.0 * sympy_inner_mat(Epos_u, Epos_v)
        I_divu_q = -sympy_div(pos_sym['u']) * pos_sym['q']
        I_divv_p = -sympy_div(pos_sym['v']) * pos_sym['p']
        return {"volume_pos_whole": simplify(_int_square(I_vol)), "divu_q_whole": simplify(_int_square(I_divu_q)), "divv_p_whole": simplify(_int_square(I_divv_p))}
    ex_whole = exact_whole_from_sympy()

    for key in ("volume_pos_whole", "divu_q_whole", "divv_p_whole"):
        E_pc = assemble_and_energy_pc(pc_whole["terms"][key], u_vec_w, v_vec_w)
        E_ng = ng_whole[key]
        import sympy as sp
        E_ex = float(sp.N(ex_whole[key]))
        err_pc_ex, (ok1, m1) = srelerr(E_pc, E_ex), verdict(srelerr(E_pc, E_ex), tol=1e-10)
        err_ng_ex, (ok2, m2) = srelerr(E_ng, E_ex), verdict(srelerr(E_ng, E_ex), tol=1e-10)
        details = f"{key:18s} | PC(dx)={E_pc:+.12e}   NG(dx)={E_ng:+.12e}   EX(dx)={E_ex:+.12e}   PC↔EX scaled.err={err_pc_ex:.3e} {m1}   NG↔EX scaled.err={err_ng_ex:.3e} {m2}"
        print(details)
        test_results.append(TestResult(f"Whole-domain: {key} PC↔EX", err_pc_ex, ok1, details))
        test_results.append(TestResult(f"Whole-domain: {key} NG↔EX", err_ng_ex, ok2, details))

    passed_tests = [r for r in test_results if r.passed]
    failed_tests = [r for r in test_results if not r.passed]
    
    print(f"\n--- Test Summary ---")
    print(f"{len(passed_tests)} tests passed, {len(failed_tests)} tests FAILED.")

    if failed_tests:
        print("\n--- Top 6 Failed Tests (by error magnitude) ---")
        failed_tests.sort(key=lambda x: x.error, reverse=True)
        for i, result in enumerate(failed_tests[:6]):
            print(f"{i+1}. {result.details}")

    print("\nDone. Green ✓ indicates success within tolerance; red ✗ highlights discrepancies.")

if __name__ == "__main__":
    main()