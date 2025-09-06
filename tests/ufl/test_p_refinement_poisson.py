# file: tests/p_ref_poisson_dh.py
import pytest
import numpy as np
import sympy as sp
import scipy.sparse.linalg as spla


from pycutfem.utils.meshgen import structured_quad, structured_triangles
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.expressions import TrialFunction, TestFunction, grad, inner, Constant, Function
from pycutfem.ufl.measures import dx
from pycutfem.ufl.forms import Equation, BoundaryCondition, assemble_form
from pycutfem.core.levelset import CircleLevelSet
import matplotlib.pyplot as plt
from pycutfem.ufl.analytic import Analytic
# ----- MMS: u = x^2 + y^2, f = -Δu = -4 --------------------------------------
x, y = sp.symbols("x y")
u_sym = sp.sin(2*sp.pi*x) * sp.sinh(sp.pi*y)   # or sp.exp(x+y), sp.sin(pi x)sin(pi y)
f_sym = -sp.diff(u_sym, x, 2) - sp.diff(u_sym, y, 2)
grad_u_sym_x = sp.diff(u_sym, x)
grad_u_sym_y = sp.diff(u_sym, y)
grad_u_x_exact = sp.lambdify((x, y), grad_u_sym_x, "numpy")
grad_u_y_exact = sp.lambdify((x, y), grad_u_sym_y, "numpy")
def vec2_callable(fx, fy):
    def f(x, y):
        ax = np.asarray(fx(x, y), dtype=float)
        ay = np.asarray(fy(x, y), dtype=float)
        # broadcast to common shape and stack along the last axis
        return np.stack([ax, ay], axis=-1)
    return f
u_exact = sp.lambdify((x, y), u_sym, "numpy")
f_exact = sp.lambdify((x, y), f_sym, "numpy")
grad_u_exact = vec2_callable(grad_u_x_exact, grad_u_y_exact)


# ----- trivial level set that selects the whole domain for side='-' ----------
L,H = 1.0, 1.0
center = (L/2, H/2)
radius = 4*max(L,H)

level_set_all = CircleLevelSet(center=center, radius=radius)

# ----- one p-run on a given element type -------------------------------------
def run_once(elem_type: str, p: int, nx=10, ny=10):
    # mesh
    if elem_type == "quad":
        nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=p)
        mesh = Mesh(nodes, element_connectivity=elems, edges_connectivity=edges,
                    elements_corner_nodes=corners, element_type="quad", poly_order=p)
    elif elem_type == "tri":
        nodes, elems, edges, corners = structured_triangles(1.0, 1.0, nx_quads=nx, ny_quads=ny, poly_order=p)
        mesh = Mesh(nodes, elems, edges, corners, element_type="tri", poly_order=p)
    else:
        raise ValueError(elem_type)

    # FE space (scalar)
    me = MixedElement(mesh, field_specs={'u': p})
    dh = DofHandler(me, method='cg')

    # boundary tag via locator (catches all high-order boundary DOFs)
    boundary = lambda X, Y: (np.isclose(X, 0.0) | np.isclose(X, 1.0) |
                             np.isclose(Y, 0.0) | np.isclose(Y, 1.0))
    dh.tag_dofs_by_locator_map({'boundary': boundary}, fields=['u'])

    # trial/test and form
    u = TrialFunction('u', dh)
    v = TestFunction ('u', dh)
    qvol = max(2*p + 2, 6)                 # conservative volume quadrature
    a = inner(grad(u), grad(v)) * dx(metadata={'q': qvol})
    L = Analytic(f_sym) * v * dx(metadata={'q': qvol})
    eq = Equation(a, L)

    # Dirichlet BC via tag
    bcs = [BoundaryCondition('u', 'dirichlet', 'boundary', lambda X, Y: u_exact(X, Y))]

    # assemble & solve
    K, F = assemble_form(eq, dof_handler=dh, bcs=bcs, quad_order=qvol, backend='python')
    uh = spla.spsolve(K, F)

    # --- L2 error over Ω using built-in
    l2 = dh.l2_error(uh, exact={'u': u_exact}, quad_order=max(2*p+4, 10), relative=False)

    # --- H1-seminorm error over Ω using the “side” helper with a constant LS
    # Build a Function view of the solution on field 'u'
    uh_fun = Function("u_h", "u", dof_handler=dh)
    fld_ids = dh.get_field_slice("u")
    uh_fun.set_nodal_values(fld_ids, uh[fld_ids])

    h1 = dh.h1_error_scalar_on_side(uh_fun, grad_u_exact,
                                    level_set=level_set_all, side='-',
                                    relative=False, quad_increase=2)
    return l2, h1

def run_suite(elem_type, orders=(1,2,3,4,5), nx=10, ny=10):
    print(f"\n== {elem_type.upper()} p-refinement on {nx}x{ny} mesh ==")
    print(f"{'p':>2}  {'L2':>12}  {'H1-semi':>12}")
    prevL2 = prevH1 = None
    l2_errors = []
    h1_errors = []
    for p in orders:
        l2, h1 = run_once(elem_type, p, nx=nx, ny=ny)
        print(f"{p:2d}  {l2:12.4e}  {h1:12.4e}")
        # if prevL2 is not None:
        #     # soft monotonicity check
        #     assert l2 <= prevL2 * 1.05 + 1e-14, "L2 did not decrease with p"
        #     assert h1 <= prevH1 * 1.05 + 1e-14, "H1 did not decrease with p"
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
def test_quad_p_refinement():
    orders = (1,2,3,4,5)
    l2_errors, h1_errors = run_suite("quad", orders=orders, nx=8,  ny=8)
    plt.figure()
    plt.loglog(orders, l2_errors, 'o-', label='L2 error')
    plt.loglog(orders, h1_errors, 's-', label='H1-semi error')
    plt.xlabel('Polynomial order p')
    plt.ylabel('Error')
    plt.title('p-refinement errors on quad mesh')
    plt.legend()
    plt.show()
    # if l2 is monotonic:
    check_monotonicity(l2_errors)
    check_monotonicity(h1_errors)

def test_tri_p_refinement():
    orders = (1,2,3,4,5)
    l2_errors, h1_errors = run_suite("tri",  orders=orders, nx=12, ny=12)
    plt.figure()
    plt.loglog(orders, l2_errors, 'o-', label='L2 error')
    plt.loglog(orders, h1_errors, 's-', label='H1-semi error')
    plt.xlabel('Polynomial order p')
    plt.ylabel('Error')
    plt.title('p-refinement errors on tri mesh')
    plt.legend()
    plt.show()
    # if l2 is monotonic:
    check_monotonicity(l2_errors)
    check_monotonicity(h1_errors)

if __name__ == "__main__":
    # use a slightly finer tri grid so p-refinement isn’t overly coarse
    run_suite("quad", orders=(1,2,3,4,5), nx=8,  ny=8)
    run_suite("tri",  orders=(1,2,3,4,5), nx=12, ny=12)

