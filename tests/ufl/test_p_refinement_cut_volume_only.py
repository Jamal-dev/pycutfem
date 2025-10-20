# tests/ufl/test_p_refinement_cut_volume_only.py
import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from pycutfem.utils.meshgen import structured_triangles
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.expressions import (
    TrialFunction, TestFunction, grad, dot,
    Neg, Pos, CellDiameter, Function, VectorFunction
)
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.measures import dx
from pycutfem.ufl.forms import Equation, assemble_form, BoundaryCondition
from pycutfem.core.levelset import AffineLevelSet

# ---- Problem setup: straight interface Γ = {x = x0}, piecewise k_± ----
k_minus, k_plus = 1.0, 10.0
x0 = 0.48
alpha_minus = 1.0
alpha_plus  = (k_minus / k_plus) * alpha_minus  # ⇒ exact flux continuity across Γ
alpha_minus = 0.0
alpha_plus = 0.0
def u_neg_xy(x, y):  # exact on Ω^-
    return np.exp(alpha_minus * (x - x0)) * np.sin(np.pi * y)

def u_pos_xy(x, y):  # exact on Ω^+
    return np.exp(alpha_plus  * (x - x0)) * np.sin(np.pi * y)

def grad_u(alpha):
    def _g(x, y):
        e = np.exp(alpha * (x - x0))
        return np.array([alpha * e * np.sin(np.pi * y), np.pi * e * np.cos(np.pi * y)], float)
    return _g

grad_u_neg = grad_u(alpha_minus)
grad_u_pos = grad_u(alpha_plus)

# manufactured forcing: -k Δu = f  with  Δu = (alpha^2 - pi^2) u
def f_side(alpha, k):
    return lambda X, Y: k*(np.pi**2 - alpha**2) * np.exp(alpha*(X - x0)) * np.sin(np.pi*Y)

f_neg_xy = f_side(alpha_minus, k_minus)
f_pos_xy = f_side(alpha_plus,  k_plus)


def solve_cut_volume_only(p: int, nx=16, ny=16):
    """
    Solve two Poisson problems on Ω^- and Ω^+ using *only* sided volume integrals (dx_neg, dx_pos).
    No interface terms are used. Returns (u_vec, dh, ls).
    """
    Lx = Ly = 1.0
    # straight-sided geometry so the cut is exact; p only changes FE order
    nodes, elems, edges, corners = structured_triangles(Lx, Ly, nx_quads=nx, ny_quads=ny, poly_order=1)
    mesh = Mesh(nodes, elems, edges, corners, element_type="tri", poly_order=1)

    # level set φ(x,y) = x - x0   (unit normal (1,0))
    ls = AffineLevelSet(1.0, 0.0, -x0)
    mesh.classify_elements(ls)
    mesh.classify_edges(ls)
    mesh.build_interface_segments(ls)

    inside_e  = mesh.element_bitset("inside")
    outside_e = mesh.element_bitset("outside")
    cut_e     = mesh.element_bitset("cut")

    has_neg   = inside_e | cut_e
    has_pos   = outside_e | cut_e

    # mixed space V_p(Ω^-) × V_p(Ω^+)
    me = MixedElement(mesh, field_specs={"u_neg": p, "u_pos": p})
    dh = DofHandler(me, method="cg")

    # deactivate wrong-side DOFs (like compressed spaces)
    dh.tag_dofs_from_element_bitset("inactive_outside", "u_neg", "outside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive_inside",  "u_pos", "inside",  strict=True)

    # outer boundary Dirichlet (exact trace) for each field
    boundary = lambda X, Y: (np.isclose(X, 0.0) | np.isclose(X, 1.0) |
                             np.isclose(Y, 0.0) | np.isclose(Y, 1.0))
    dh.tag_dofs_by_locator_map({'boundary': boundary}, fields=['u_neg','u_pos'])

    # sided trial/test
    u_neg = TrialFunction("u_neg", dh, side="-"); v_neg = TestFunction("u_neg", dh, side="-")
    u_pos = TrialFunction("u_pos", dh, side="+"); v_pos = TestFunction("u_pos", dh, side="+")

    # measures: *volume only*, split by side; no interface measure used here
    qvol = max(2*p + 2, 8)
    dx_neg = dx(defined_on=has_neg, level_set=ls, metadata={"side":"-", "q": qvol})
    dx_pos = dx(defined_on=has_pos, level_set=ls, metadata={"side":"+", "q": qvol})

    # bilinear (volume only)
    a = (
        k_minus*dot(grad(Neg(u_neg)), grad(Neg(v_neg))) * dx_neg +
        k_plus *dot(grad(Pos(u_pos)), grad(Pos(v_pos))) * dx_pos
    )

    # RHS (manufactured loads)
    F = Analytic(f_neg_xy)*v_neg*dx_neg + Analytic(f_pos_xy)*v_pos*dx_pos

    bcs = [
        BoundaryCondition('u_neg','dirichlet','boundary', u_neg_xy),
        BoundaryCondition('u_pos','dirichlet','boundary', u_pos_xy),
        BoundaryCondition('u_pos','dirichlet','inactive_inside',  0.0),
        BoundaryCondition('u_neg','dirichlet','inactive_outside', 0.0),
    ]

    K, rhs = assemble_form(Equation(a, F), dof_handler=dh, bcs=bcs, backend='python')
    assert np.isfinite(K.data).all() and np.isfinite(rhs).all(), "NaN/Inf in volume-only assembly"
    u_vec = spsolve(K, rhs)
    assert np.isfinite(u_vec).all(), "NaN/Inf in volume-only solve"
    return u_vec, dh, ls


def h1_errors_on_sides(u_vec, dh: DofHandler, ls):
    """
    Use DofHandler's H1 error helpers to compute piecewise H1-seminorm errors on Ω^- and Ω^+.
    """
    # pack solution into a VectorFunction and slice back components
    U = VectorFunction("U", ["u_neg", "u_pos"], dof_handler=dh)
    for fld in ["u_neg", "u_pos"]:
        sl = dh.get_field_slice(fld)
        U.set_nodal_values(sl, u_vec[sl])

    uh_neg = Function("uh_neg", "u_neg", dof_handler=dh, parent_vector=U, component_index=0)
    uh_pos = Function("uh_pos", "u_pos", dof_handler=dh, parent_vector=U, component_index=1)

    eH1_neg = dh.h1_error_scalar_on_side(uh_neg, grad_u_neg, ls, side='-', relative=False)
    eH1_pos = dh.h1_error_scalar_on_side(uh_pos, grad_u_pos, ls, side='+', relative=False)
    eH1_tot = (eH1_neg**2 + eH1_pos**2)**0.5
    return eH1_neg, eH1_pos, eH1_tot


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_cut_volume_only_h1_by_p(p):
    u_vec, dh, ls = solve_cut_volume_only(p)
    eH1_neg, eH1_pos, eH1_tot = h1_errors_on_sides(u_vec, dh, ls)
    print(f"[p={p}]  H1-: {eH1_neg:.3e}  H1+: {eH1_pos:.3e}  H1tot: {eH1_tot:.3e}")


def test_cut_volume_only_convergence_overall():
    p_list = [1, 2, 3, 4]
    errs = []
    for p in p_list:
        u_vec, dh, ls = solve_cut_volume_only(p)
        eH1_neg, eH1_pos, eH1_tot = h1_errors_on_sides(u_vec, dh, ls)
        errs.append((p, eH1_neg, eH1_pos, eH1_tot))

    for p, eN, eP, eT in errs:
        print(f"[p={p}]  H1-={eN:.3e}  H1+={eP:.3e}  H1tot={eT:.3e}")

    # Guard: overall decrease from p=1 to p=max
    eN1, eP1, eT1 = errs[0][1], errs[0][2], errs[0][3]
    eNL, ePL, eTL = errs[-1][1], errs[-1][2], errs[-1][3]
    assert eNL < eN1, "Ω^- H1 error did not decrease overall with p."
    assert ePL < eP1, "Ω^+ H1 error did not decrease overall with p."
    assert eTL < eT1, "Total H1 error did not decrease overall with p."

    # Near-monotone step-to-step (tolerate tiny wiggles)
    def nonincreasing(seq, tol=5e-3):
        return all(seq[i+1] <= seq[i]*(1+tol) for i in range(len(seq)-1))

    assert nonincreasing([e for _, e, _, _ in errs]),  "Ω^- H1 not nonincreasing."
    assert nonincreasing([e for _, _, e, _ in errs]),  "Ω^+ H1 not nonincreasing."
    assert nonincreasing([e for _, _, _, e in errs]),  "Total H1 not nonincreasing."

