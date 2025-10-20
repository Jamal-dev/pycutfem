# tests/ufl/test_p_refinement_interface_full.py
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pycutfem.utils.meshgen import structured_triangles
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.expressions import (
    TrialFunction, TestFunction, grad, dot,
    Pos, Neg, FacetNormal, CellDiameter, Constant,
    VectorFunction, Function
)
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.measures import dx, dInterface
from pycutfem.ufl.forms import Equation, assemble_form, BoundaryCondition
from pycutfem.core.levelset import AffineLevelSet

# ----- Problem setup: Γ = {x = x0}, k_- and k_+ piecewise constants -----
k_minus, k_plus = 1.0, 10.0
x0 = 0.48
alpha_minus = 1.0
alpha_plus  = (k_minus / k_plus) * alpha_minus   # ensures flux continuity in exact solution

def u_neg_xy(x, y):  # exact in Ω^-
    return np.exp(alpha_minus * (x - x0)) * np.sin(np.pi * y)

def u_pos_xy(x, y):  # exact in Ω^+
    return np.exp(alpha_plus  * (x - x0)) * np.sin(np.pi * y)

def grad_u(alpha):
    """∇u for u(x,y) = exp(alpha*(x-x0))*sin(pi*y)."""
    def _g(x, y):
        e = np.exp(alpha * (x - x0))
        return np.array([alpha * e * np.sin(np.pi * y), np.pi * e * np.cos(np.pi * y)], dtype=float)
    return _g

grad_u_neg = grad_u(alpha_minus)
grad_u_pos = grad_u(alpha_plus)

def f_side(alpha, k):
    """Manufactured forcing for -k Δu = f, with u above; Δu=(alpha^2 - pi^2)u."""
    return lambda X, Y: k * (np.pi**2 - alpha**2) * np.exp(alpha * (X - x0)) * np.sin(np.pi * Y)

f_neg_xy = f_side(alpha_minus, k_minus)
f_pos_xy = f_side(alpha_plus,  k_plus)


def solve_once(p: int, nx=16, ny=16):
    """Solve the interface problem with symmetric Nitsche; return (u_vec, dh, level_set, cut_e)."""
    # -- Mesh: straight-sided geometry; p only affects FE order
    Lx = Ly = 1.0
    nodes, elems, edges, corners = structured_triangles(Lx, Ly, nx_quads=nx, ny_quads=ny, poly_order=1)
    mesh = Mesh(nodes, elems, edges, corners, element_type="tri", poly_order=1)

    # Level set φ(x,y)=x-x0
    ls = AffineLevelSet(1.0, 0.0, -x0).normalised()
    mesh.classify_elements(ls)
    mesh.classify_edges(ls)
    mesh.build_interface_segments(ls)

    inside_e  = mesh.element_bitset("inside")
    outside_e = mesh.element_bitset("outside")
    cut_e     = mesh.element_bitset("cut")
    has_neg   = inside_e | cut_e
    has_pos   = outside_e | cut_e

    # Mixed FE space
    me = MixedElement(mesh, field_specs={"u_neg": p, "u_pos": p})
    dh = DofHandler(me, method="cg")

    # Deactivate wrong-side DOFs (keep system well-posed and physical)
    dh.tag_dofs_from_element_bitset("inactive_outside", "u_neg", "outside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive_inside",  "u_pos", "inside",  strict=True)

    # Dirichlet on outer boundary for each field (exact trace)
    boundary = lambda X, Y: (np.isclose(X, 0.0) | np.isclose(X, 1.0) |
                             np.isclose(Y, 0.0) | np.isclose(Y, 1.0))
    dh.tag_dofs_by_locator_map({'boundary': boundary}, fields=['u_neg', 'u_pos'])

    # Trial/test (sided)
    u_neg = TrialFunction("u_neg", dh, side="-"); v_neg = TestFunction("u_neg", dh, side="-")
    u_pos = TrialFunction("u_pos", dh, side="+"); v_pos = TestFunction("u_pos", dh, side="+")

    # Measures
    qvol  = max(2*p + 2, 8)
    qedge = max(2*p + 4, 10)
    dx_neg = dx(defined_on=has_neg, level_set=ls, metadata={"side": "-", "q": qvol})
    dx_pos = dx(defined_on=has_pos, level_set=ls, metadata={"side": "+", "q": qvol})
    dG     = dInterface(defined_on=cut_e, level_set=ls, metadata={"q": qedge})

    # Symmetric Nitsche coupling on Γ
    n  = FacetNormal()
    h  = CellDiameter()
    def tr(w): return dot(grad(w), n)

    avg_flux_trial = 0.5*(k_plus*tr(Pos(u_pos)) + k_minus*tr(Neg(u_neg)))
    avg_flux_test  = 0.5*(k_plus*tr(Pos(v_pos)) + k_minus*tr(Neg(v_neg)))
    jump_u = Pos(u_pos) - Neg(u_neg)
    jump_v = Pos(v_pos) - Neg(v_neg)

    beta = 20.0
    lam  = Constant(beta * max(k_minus, k_plus) * (p+1)**2) / h

    a = (
        k_minus*dot(grad(Neg(u_neg)), grad(Neg(v_neg)))*dx_neg +
        k_plus *dot(grad(Pos(u_pos)), grad(Pos(v_pos)))*dx_pos +
        ( avg_flux_trial*jump_v + avg_flux_test*jump_u + lam*jump_u*jump_v )*dG
    )

    # RHS
    F = Analytic(f_neg_xy)*v_neg*dx_neg + Analytic(f_pos_xy)*v_pos*dx_pos

    bcs = [
        BoundaryCondition('u_neg','dirichlet','boundary', u_neg_xy),
        BoundaryCondition('u_pos','dirichlet','boundary', u_pos_xy),
        BoundaryCondition('u_pos','dirichlet','inactive_inside',  0.0),
        BoundaryCondition('u_neg','dirichlet','inactive_outside', 0.0),
    ]

    K, rhs = assemble_form(Equation(a, F), dof_handler=dh, bcs=bcs, backend='python')
    assert np.isfinite(K.data).all() and np.isfinite(rhs).all(), "NaN/Inf in assembled system"
    u_vec = spsolve(K, rhs)
    assert np.isfinite(u_vec).all(), "NaN/Inf in solution vector"

    return u_vec, dh, ls, cut_e


def interface_energies(u_vec, dh: DofHandler, ls, cut_e, p):
    """Compute unscaled jump and flux-jump energies on Γ for a solved u_vec."""
    # Reuse the same sided functions and measures
    u_neg = TrialFunction("u_neg", dh, side="-"); v_neg = TestFunction("u_neg", dh, side="-")
    u_pos = TrialFunction("u_pos", dh, side="+"); v_pos = TestFunction("u_pos", dh, side="+")

    qedge = max(2*p + 4, 10)
    dG    = dInterface(defined_on=cut_e, level_set=ls, metadata={"q": qedge})
    n     = FacetNormal()
    def tr(w): return dot(grad(w), n)

    # Unscaled jump energy matrix
    jump_u = Pos(u_pos) - Neg(u_neg)
    jump_v = Pos(v_pos) - Neg(v_neg)
    a_jump = (jump_u * jump_v) * dG

    # Flux-jump energy matrix
    flux_u = k_plus*tr(Pos(u_pos)) - k_minus*tr(Neg(u_neg))
    flux_v = k_plus*tr(Pos(v_pos)) - k_minus*tr(Neg(v_neg))
    a_flux = (flux_u * flux_v) * dG

    K_jump, _ = assemble_form(Equation(a_jump, 0.0), dof_handler=dh, backend="python")
    K_flux, _ = assemble_form(Equation(a_flux, 0.0), dof_handler=dh, backend="python")

    assert np.isfinite(K_jump.data).all() and np.isfinite(K_flux.data).all(), "NaN/Inf in Γ matrices"

    Ejump = float(u_vec.T @ (K_jump @ u_vec))
    Eflux = float(u_vec.T @ (K_flux @ u_vec))
    return Ejump, Eflux


def h1_errors_piecewise(u_vec, dh: DofHandler, ls):
    """Compute H1-seminorm errors on each side using your DofHandler helpers."""
    # Wrap solution into a VectorFunction and populate nodal values
    U = VectorFunction("U", ["u_neg", "u_pos"], dof_handler=dh)
    # Fill from the global solution
    for fld in ["u_neg", "u_pos"]:
        sl = dh.get_field_slice(fld)
        U.set_nodal_values(sl, u_vec[sl])

    # Scalar Function views for each field
    uh_neg = Function("uh_neg", "u_neg", dof_handler=dh, parent_vector=U, component_index=0)
    uh_pos = Function("uh_pos", "u_pos", dof_handler=dh, parent_vector=U, component_index=1)

    # H1-seminorm errors on each side
    eH1_neg = dh.h1_error_scalar_on_side(uh_neg, grad_u_neg, ls, side='-', relative=False)
    eH1_pos = dh.h1_error_scalar_on_side(uh_pos, grad_u_pos, ls, side='+', relative=False)
    eH1_tot = (eH1_neg**2 + eH1_pos**2)**0.5
    return eH1_neg, eH1_pos, eH1_tot


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_interface_p_refinement_jump_flux_and_h1(p):
    u_vec, dh, ls, cut_e = solve_once(p)
    Ejump, Eflux = interface_energies(u_vec, dh, ls, cut_e, p)
    eH1_neg, eH1_pos, eH1_tot = h1_errors_piecewise(u_vec, dh, ls)
    print(f"[p={p}]  E_jump={Ejump:.3e}  E_flux={Eflux:.3e}  "
          f"H1-: {eH1_neg:.3e}  H1+: {eH1_pos:.3e}  H1tot: {eH1_tot:.3e}")

# A simple convergence guard across p
def test_interface_convergence_overall():
    p_list = [1, 2, 3, 4]
    vals = []
    for p in p_list:
        u_vec, dh, ls, cut_e = solve_once(p)
        Ejump, Eflux = interface_energies(u_vec, dh, ls, cut_e, p)
        eH1_neg, eH1_pos, eH1_tot = h1_errors_piecewise(u_vec, dh, ls)
        vals.append((p, Ejump, Eflux, eH1_tot))
    # Print for inspection
    for p, Ej, Ef, Eh in vals:
        print(f"[p={p}]  E_jump={Ej:.3e}  E_flux={Ef:.3e}  H1tot={Eh:.3e}")

    # Require overall decrease from p=1 to p=max
    Ej1, Ef1, Eh1 = vals[0][1], vals[0][2], vals[0][3]
    EjL, EfL, EhL = vals[-1][1], vals[-1][2], vals[-1][3]
    assert EjL < Ej1, "Jump energy did not decrease overall with p."
    assert EfL < Ef1, "Flux-jump energy did not decrease overall with p."
    assert EhL < Eh1, "H1-seminorm error did not decrease overall with p."
