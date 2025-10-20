import pytest
import numpy as np
import scipy.sparse.linalg as spla

from pycutfem.utils.meshgen import structured_quad, structured_triangles
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.expressions import (
    TrialFunction, TestFunction, grad, inner, dot,
    Pos, Neg, FacetNormal, CellDiameter, Constant, Function
)
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.measures import dx, dInterface
from pycutfem.ufl.forms import Equation, BoundaryCondition, assemble_form
# Import your Affine LS implementation (unit normal supported)
from pycutfem.core.levelset import AffineLevelSet  # -> φ(x,y)=a x + b y + c  :contentReference[oaicite:1]{index=1}
import matplotlib.pyplot as plt
from pycutfem.io.visualization import plot_mesh_2

# --- piecewise coefficients & exact fields on x<0 (−) and x>0 (+) -------------
k_minus, k_plus = 1.0, 10.0
alpha_minus = 1.0
alpha_plus  = (k_minus/k_plus)*alpha_minus
x0 = 0.48

def u_neg_xy(x,y): return np.exp(alpha_minus*(x - x0)) * np.sin(np.pi*y)
def u_pos_xy(x,y): return np.exp(alpha_plus *(x - x0)) * np.sin(np.pi*y)

def f_neg_xy(x,y):
    return k_minus*(np.pi**2 - alpha_minus**2) * np.exp(alpha_minus*(x - x0)) * np.sin(np.pi*y)

def f_pos_xy(x,y):
    return k_plus *(np.pi**2 - alpha_plus **2) * np.exp(alpha_plus *(x - x0)) * np.sin(np.pi*y)

def grad_u_neg(x,y):
    e = np.exp(alpha_minus*(x - x0))
    return np.array([alpha_minus*e*np.sin(np.pi*y), np.pi*e*np.cos(np.pi*y)])

def grad_u_pos(x,y):
    e = np.exp(alpha_plus*(x - x0))
    return np.array([alpha_plus*e*np.sin(np.pi*y), np.pi*e*np.cos(np.pi*y)])

# --- one run (elem_type in {"quad","tri"}) ------------------------------------
def run_once_interface(elem_type: str, p: int, nx=14, ny=14):
    Lx = Ly = 1.0
    if elem_type == "quad":
        nodes, elems, edges, corners = structured_quad(Lx, Ly, nx=nx, ny=ny, poly_order=p)
        mesh = Mesh(nodes, element_connectivity=elems, edges_connectivity=edges,
                    elements_corner_nodes=corners, element_type="quad", poly_order=p)
    else:
        nodes, elems, edges, corners = structured_triangles(Lx, Ly, nx_quads=nx, ny_quads=ny, poly_order=p)
        mesh = Mesh(nodes, elems, edges, corners, element_type="tri", poly_order=p)

    # Level set φ(x,y)=x: interface Γ={x=0}. Use unit-normalised version.
    ls = AffineLevelSet(1.0, 0.0, -x0).normalised()  # n = (1,0) everywhere  
    mesh.classify_elements(ls)
    mesh.classify_edges(ls)
    mesh.build_interface_segments(ls)

    # BitSets / regions
    inside_e  = mesh.element_bitset("inside")   # φ<0 : x<0 (NEG side)
    outside_e = mesh.element_bitset("outside")  # φ>0 : x>0 (POS side)
    cut_e     = mesh.element_bitset("cut")
    has_neg   = inside_e | cut_e
    has_pos   = outside_e | cut_e

    # Mixed scalar fields for the two subdomains
    me = MixedElement(mesh, field_specs={'u_neg': p, 'u_pos': p})
    dh = DofHandler(me, method='cg')
    plot_mesh_2(mesh, level_set=ls)

    # Strong Dirichlet on the *outer* boundary for each field (exact trace)
    boundary = lambda X,Y: (np.isclose(X,0.0) | np.isclose(X,1.0) |
                            np.isclose(Y,0.0) | np.isclose(Y,1.0))
    dh.tag_dofs_by_locator_map({'boundary': boundary}, fields=['u_neg','u_pos'])

    # Deactivate wrong-side DOFs (like NGSolve compressed spaces)
    dh.tag_dofs_from_element_bitset("inactive_inside",  "u_pos", "inside",  strict=True)
    dh.tag_dofs_from_element_bitset("inactive_outside", "u_neg", "outside", strict=True)

    # Trial/test and measures
    u_neg = TrialFunction('u_neg', dh, side='-'); v_neg = TestFunction('u_neg', dh, side='-')
    u_pos = TrialFunction('u_pos', dh, side='+'); v_pos = TestFunction('u_pos', dh, side='+')

    qvol  = max(2*p+2, 6)
    qedge = max(2*p+4, 8)
    dx_neg = dx(defined_on=has_neg, level_set=ls, metadata={'side':'-','q':qvol})
    dx_pos = dx(defined_on=has_pos, level_set=ls, metadata={'side':'+','q':qvol})
    dG     = dInterface(defined_on=cut_e, level_set=ls, metadata={'q':qedge})
    n      = FacetNormal()
    h      = CellDiameter()

    # Volume terms
    a = k_minus*inner(grad(Neg(u_neg)), grad(Neg(v_neg)))*dx_neg \
      + k_plus *inner(grad(Pos(u_pos)), grad(Pos(v_pos)))*dx_pos

    # Symmetric Nitsche on Γ: average flux + penalty
    def tr(u): return dot(grad(u), n)
    avg_flux_trial = 0.5*(k_plus*tr(Pos(u_pos)) + k_minus*tr(Neg(u_neg)))
    avg_flux_test  = 0.5*(k_plus*tr(Pos(v_pos)) + k_minus*tr(Neg(v_neg)))
    jump_u = Pos(u_pos) - Neg(u_neg)
    jump_v = Pos(v_pos) - Neg(v_neg)

    beta = 20.0
    lam  = Constant(beta * max(k_minus, k_plus) * (p+1)**2) / h
    a   += ( avg_flux_trial*jump_v + avg_flux_test*jump_u + lam*jump_u*jump_v ) * dG

    # Piecewise RHS (no placeholders)
    f = Analytic(f_neg_xy)*v_neg*dx_neg + Analytic(f_pos_xy)*v_pos*dx_pos
    eq = Equation(a, f)

    # Dirichlet values and zero for inactive tags
    bcs = [
        BoundaryCondition('u_neg','dirichlet','boundary', lambda X,Y: u_neg_xy(X,Y)),
        BoundaryCondition('u_pos','dirichlet','boundary', lambda X,Y: u_pos_xy(X,Y)),
        BoundaryCondition('u_pos','dirichlet','inactive_inside',  0.0),
        BoundaryCondition('u_neg','dirichlet','inactive_outside', 0.0),
    ]

    K, F = assemble_form(eq, dof_handler=dh, bcs=bcs, quad_order=qvol, backend='python')
    assert np.isfinite(K.data).all(), "NaN/Inf in K from interface path"
    assert np.isfinite(F).all(),      "NaN/Inf in F from interface path"
    sol = spla.spsolve(K, F)
    # rank of K matrix
    rank = np.linalg.matrix_rank(K.toarray())
    assert rank == K.shape[0], f"Singular matrix K (rank {rank} < {K.shape[0]})"
    assert np.isfinite(sol).all(),    "NaN/Inf in sol from interface path"

    # Build Function views and load the solution
    uh_neg = Function("u_neg", field_name='u_neg', dof_handler=dh, side='-')
    uh_pos = Function("u_pos", field_name='u_pos', dof_handler=dh, side='+')
    dh.add_to_functions(sol, [uh_neg, uh_pos])

    # Errors on each side; sum for a global indicator
    qerr = max(2*p+4, 10)

    # --- L2 on Ω− and Ω+ (pass dicts for exact)
    l2m = dh.l2_error_on_side(
        uh_neg, side='-', level_set=ls, quad_order=qerr,
        exact={'u_neg': u_neg_xy},
    )
    l2p = dh.l2_error_on_side(
        uh_pos, side='+', level_set=ls, quad_order=qerr,
        exact={'u_pos': u_pos_xy},
    )

    # --- H1-semi on Ω− and Ω+ (pass dicts for exact_grad)
    h1m = dh.h1_error_scalar_on_side(
        uh_neg, side='-', level_set=ls, quad_increase=2,
        exact_grad= grad_u_neg,
    )
    h1p = dh.h1_error_scalar_on_side(
        uh_pos, side='+', level_set=ls, quad_increase=2,
        exact_grad= grad_u_pos,
    )

    return (l2m + l2p, h1m + h1p)

def run_suite_interface(elem_type, orders=(1,2,3,4,5), nx=14, ny=14):
    print(f"\n== Interface p-refinement ({elem_type}) ==")
    print(f"{'p':>2}  {'L2':>12}  {'H1-semi':>12}")
    prevL2=prevH1=None
    l2_errors = []
    h1_errors = []
    for p in orders:
        l2, h1 = run_once_interface(elem_type, p, nx=nx, ny=ny)
        print(f"{p:2d}  {l2:12.4e}  {h1:12.4e}")
        # if prevL2 is not None:
        #     assert l2 <= prevL2*1.2 + 1e-14
        #     assert h1 <= prevH1*1.2 + 1e-14
        # prevL2, prevH1 = l2, h1
        l2_errors.append(l2)
        h1_errors.append(h1)
    return l2_errors, h1_errors

def check_monotonicity(errors):
    prev = None
    for e in errors:
        if prev is not None:
            assert e <= prev * 1.05 + 1e-14, "Error did not decrease with p"
        prev = e
def test_tri_p_refinement_interface():
    orders = (1,2,3)
    l2_errors, h1_errors = run_suite_interface("tri",  orders=orders, nx=16, ny=16)
    plt.figure()
    plt.loglog(orders, l2_errors, 'o-', label='L2 error')
    plt.loglog(orders, h1_errors, 's-', label='H1-semi error')
    plt.xlabel('Polynomial order p')
    plt.ylabel('Error')
    plt.title('p-refinement errors on tri mesh')
    plt.legend()
    plt.show()
    check_monotonicity(l2_errors)
    check_monotonicity(h1_errors)
def test_quad_p_refinement_interface():
    orders = (1,2,3)
    l2_errors, h1_errors = run_suite_interface("quad", orders=orders, nx=12, ny=12)
    plt.figure()
    plt.loglog(orders, l2_errors, 'o-', label='L2 error')
    plt.loglog(orders, h1_errors, 's-', label='H1-semi error')
    plt.xlabel('Polynomial order p')
    plt.ylabel('Error')
    plt.title('p-refinement errors on quad mesh')
    plt.legend()
    plt.show()
    check_monotonicity(l2_errors)
    check_monotonicity(h1_errors)

if __name__ == "__main__":
    run_suite_interface("quad", orders=(1,2,3,4,5), nx=12, ny=12)
    run_suite_interface("tri",  orders=(1,2,3,4,5), nx=16, ny=16)

