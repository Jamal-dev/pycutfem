# tests/test_ghost_edge_penalty.py  – CutFEM ghost‐edge verification suite
# =======================================================================
# The cases below exercise **only** the ghost‐edge assembler.  They are
# intentionally light‑weight (O(10² dofs)) so they can run in a few seconds in
# CI while still catching sign, orientation, and higher‑derivative issues.
#
# PyTest discovers this file automatically.

import numpy as np
import pytest

# --- core imports -----------------------------------------------------------
from pycutfem.core.mesh import Mesh
from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.levelset import LevelSetFunction, CircleLevelSet, AffineLevelSet
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.domain_manager import get_domain_bitset

# --- UFL / compiler ----------------------------------------------------------
from pycutfem.ufl.expressions import (Function, TrialFunction, 
                                      TestFunction, Derivative, 
                                      Constant, Jump, 
                                      VectorTrialFunction, 
                                      VectorTestFunction,
                                      Hessian, FacetNormal, inner, dot)
from pycutfem.ufl.measures import dGhost, dx
from pycutfem.ufl.compilers import FormCompiler
from tests.ufl.test_face_integrals import dof_handler
from pycutfem.ufl.forms import BoundaryCondition, assemble_form, Equation
from pycutfem.io.visualization import plot_mesh_2
import matplotlib.pyplot as plt

X_IFACE = 1.03  # avoid mesh-aligned interface so ghost edges are non-empty


# Lx,Ly = 2.0, 1.0
# level_set = CircleLevelSet(radius=0.5, center=(Lx/2, Ly/2))
# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def hessian_inner(u, v):
    return inner(Hessian(u), Hessian(v)) 


def hdotn(expr):
    """Convenience: (Hessian(expr)) · n  (vector in R^2)."""
    n = FacetNormal()
    return dot(Hessian(expr), n)


def _edge_length_sum(mesh, bitset):
    length = 0.0
    for eid in bitset.to_indices():
        e = mesh.edge(int(eid))
        p0, p1 = mesh.nodes_x_y_pos[list(e.nodes)]
        length += float(np.linalg.norm(p1 - p0))
    return length


def _edge_length_weighted(mesh, bitset, component: int):
    length = 0.0
    for eid in bitset.to_indices():
        e = mesh.edge(int(eid))
        p0, p1 = mesh.nodes_x_y_pos[list(e.nodes)]
        seg_len = float(np.linalg.norm(p1 - p0))
        length += float(e.normal[component] ** 2) * seg_len
    return length

def _q2_second_derivative_1d(v0, v1, v2, h):
    return 4.0 * (v0 - 2.0 * v1 + v2) / (h * h)


def _cut_column_positions(mesh, x_iface):
    corner_nodes = np.asarray(mesh.corner_connectivity).ravel()
    xs = np.unique(mesh.nodes_x_y_pos[corner_nodes, 0])
    xs.sort()
    idx = int(np.searchsorted(xs, x_iface) - 1)
    if idx < 0 or idx + 1 >= xs.size:
        raise ValueError("Interface not inside mesh bounds.")
    x_left = float(xs[idx])
    x_right = float(xs[idx + 1])
    x_mid = 0.5 * (x_left + x_right)
    return x_left, x_mid, x_right, x_right - x_left


def _cut_hxx(mesh, x_iface):
    x_left, x_mid, x_right, h_x = _cut_column_positions(mesh, x_iface)
    def pos_val(x):
        return (x - x_iface) ** 2 if x > x_iface else 0.0
    def neg_val(x):
        return 0.0 if x > x_iface else (x - x_iface) ** 2
    hxx_pos = _q2_second_derivative_1d(pos_val(x_left), pos_val(x_mid), pos_val(x_right), h_x)
    hxx_neg = _q2_second_derivative_1d(neg_val(x_left), neg_val(x_mid), neg_val(x_right), h_x)
    return hxx_pos, hxx_neg



@pytest.fixture(scope="module")
def setup_quad2():
    """2×1 quadratic mesh cut vertically at x=X_IFACE."""
    poly_order = 2
    L, H = 2.0, 1.0
    nx, ny = 20, 5
    (nodes, elements_connectivity, 
     edge_connectivity, corner_nodes) = structured_quad(L, 
                                                        H, 
                                                        nx=nx, 
                                                        ny=ny, 
                                                        poly_order=poly_order)
    mesh = Mesh(nodes = nodes,
                element_connectivity = elements_connectivity,
                edges_connectivity = edge_connectivity,
                elements_corner_nodes = corner_nodes,
                element_type="quad",
                poly_order=poly_order)
    
    level_set = AffineLevelSet(a=1.0, b=0, c=-X_IFACE)  # Vertical line at x=X_IFACE

    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)

    ghost = mesh.edge_bitset('ghost')
    # fig, ax = plt.subplots(figsize=(10, 8))
    # plot_mesh_2(mesh, ax=ax, level_set=level_set, show=True, 
    #           plot_nodes=False, elem_tags=True, edge_colors=True)
    assert ghost.cardinality() > 0, "Mesh should contain ghost edges for the test."
    # This call now works because of the fixes in visualization.py
    

    me = MixedElement(mesh, field_specs={"u": poly_order})
    dh = DofHandler(me, method="cg")

    comp = FormCompiler(dh, quadrature_order=4)

    return mesh, level_set, ghost, dh, comp

# ---------------------------------------------------------------------------
# 1. Structural check – SPD + symmetry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", [ "jit", "python"])
def test_hessian_penalty_spd(setup_quad2, backend):
    _mesh, ls, ghost, dh, comp = setup_quad2
    u_pos = TrialFunction(field_name="u",name="u_trial_pos",dof_handler=dh, side="+") 
    v_pos = TestFunction( field_name="u",name="v_test_pos",dof_handler=dh, side="+")
    u_neg = TrialFunction(field_name="u",name="u_trial_neg",dof_handler=dh, side="-") 
    v_neg = TestFunction( field_name="u",name="v_test_neg",dof_handler=dh, side="-")

    a = hessian_inner(Jump(u_pos,u_neg), Jump(v_pos,v_neg)) * dGhost(defined_on=ghost, 
                                                 level_set=ls, 
                                                 metadata={"derivs": {(2,0),(1,1),(0,2)}, "q":6})
    A,_ = assemble_form(Equation(a,None), dof_handler=dh, bcs=[], backend=backend)
    K = A.toarray()
    # print("Hessian penalty matrix K:\n", K )

    # symmetric
    assert np.allclose(K, K.T, atol=1e-12)
    # positive semi‑definite
    evals = np.linalg.eigvalsh(K)
    lam_min = evals[0]
    lam_max = evals[-1]
    eps = np.finfo(float).eps

    # Robust PSD check (relative to the matrix scale)
    tol = 200 * eps * max(1.0, lam_max)   # ~ O(1e-8) here
    assert lam_min >= -tol


# ---------------------------------------------------------------------------
# 2. Zero‑jump check – quadratic function (constant Hessian)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", [ "jit", "python"])
def test_hessian_energy_zero_for_quadratic(setup_quad2, backend):
    _mesh, ls, ghost, dh, comp = setup_quad2

    uh = Function(name="uh", field_name="u", dof_handler=dh)
    uh.set_values_from_function(lambda x, y: x**2 + y**2)  # constant Hessian
    jump_u = Jump(uh)
    energy_form = hessian_inner(jump_u, jump_u) * dGhost(defined_on=ghost, level_set=ls, metadata={"q":4})
    F = None  # dummy lhs; we need scalar assembly path
    assembler_hooks={type(energy_form.integrand):{'name':'E'}}
    res = assemble_form(Equation(F, energy_form),  dof_handler=dh, bcs=[], assembler_hooks=assembler_hooks, backend=backend)
    assert abs(res["E"]) < 1e-12

# ---------------------------------------------------------------------------
# 3. Analytic value – manufactured jump 2 on Hessian xx
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", [ "jit", "python"])
def test_hessian_energy_known_value(setup_quad2, backend):
    _mesh, ls, ghost, dh, comp = setup_quad2

    def piecewise_pos(x, y):
        return np.where(x > X_IFACE, (x - X_IFACE) ** 2, 0.0)
    def piecewise_neg(x, y):
        return np.where(x > X_IFACE, 0.0, (x - X_IFACE) ** 2)
    u_pos = Function(name ="u", field_name="u", dof_handler=dh)
    u_pos.set_values_from_function(piecewise_pos)
    u_neg = Function(name ="u", field_name="u", dof_handler=dh)
    u_neg.set_values_from_function(piecewise_neg)
    jump_u = Jump(u_pos, u_neg)

    energy_form = hessian_inner(jump_u, jump_u) * dGhost(defined_on=ghost, level_set=ls, metadata={"q":4})
    assembler_hooks={type(energy_form.integrand):{'name':'E'}}
    F = None
    res = assemble_form(Equation(F, energy_form),  dof_handler=dh, bcs=[], assembler_hooks=assembler_hooks, backend=backend)

    assembled = res["E"]
    ghost_pos = _mesh.edge_bitset("ghost_pos")
    ghost_neg = _mesh.edge_bitset("ghost_neg")
    ghost_both = _mesh.edge_bitset("ghost_both")
    hxx_pos_cut, hxx_neg_cut = _cut_hxx(_mesh, X_IFACE)
    expected = (
        (2.0 - hxx_neg_cut) ** 2 * _edge_length_sum(_mesh, ghost_pos)
        + (hxx_pos_cut - 2.0) ** 2 * _edge_length_sum(_mesh, ghost_neg)
        + (hxx_pos_cut - hxx_neg_cut) ** 2 * _edge_length_sum(_mesh, ghost_both)
    )
    assert np.isclose(assembled, expected, rtol=1e-10, atol=1e-12)

@pytest.mark.parametrize("backend", [ "jit", "python"])
def test_scalar_jump_penalty_spd(setup_quad2, backend):
    """Ghost penalty K = ∫_Γ ⟦u⟧⟦v⟧ ds must be symmetric PSD."""
    _mesh, level_set, ghost_domain, dh, compiler = setup_quad2

    u = TrialFunction(field_name="u",name="u",dof_handler=dh) 
    v = TestFunction( field_name="u",name="v",dof_handler=dh)
    a = Jump(u) * Jump(v) * dGhost(defined_on=ghost_domain, level_set=level_set, metadata={"q":4})

    K,_ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend=backend)
    Kd = K.toarray()
    assert np.allclose(Kd, Kd.T), 'K not symmetric'
    eig = np.linalg.eigvalsh(Kd)
    assert np.all(eig >= -1e-12), 'K not PSD'

### Test 4: Mathematical Exactness (Zero-Jump) ###
@pytest.mark.parametrize("backend", [ "jit", "python"])
def test_hessian_penalty_exactness_for_quadratics(setup_quad2, backend):
    """
    Tests that the penalty energy is zero for a function whose Hessian jump is zero.
    For u(x,y) = x², the Hessian is constant, so its jump is zero.
    """
    _mesh, level_set, ghost_domain, dh, compiler = setup_quad2
    
    u_h = Function('u', 'u', dh)
    u_h.set_values_from_function(lambda x, y: x**2)

    penalty_form = hessian_inner(Jump(u_h), Jump(u_h)) * dGhost(defined_on=ghost_domain, level_set=level_set, metadata={"q":4})
    
    # Assemble the scalar functional
    assembler_hooks={type(penalty_form.integrand):{'name':'penalty_energy'}}
    F = None
    res = assemble_form(Equation(F, penalty_form),  
                        dof_handler=dh, bcs=[], 
                        assembler_hooks=assembler_hooks, 
                        backend=backend)
    assembled_energy = res['penalty_energy']

    assert abs(assembled_energy) < 1e-12

### Test 3: Quantitative Correctness (Constant-Jump) ###
@pytest.mark.parametrize("backend", [ "jit", "python"])
def test_hessian_penalty_quantitative_value(setup_quad2, backend):
    """
    Tests that the assembled value of the Hessian penalty for a manufactured
    solution matches the known analytical value.
    """
    _mesh, level_set, ghost_domain, dh, compiler = setup_quad2

    # We use a function u that is C¹ continuous, but whose second derivative jumps.
    # u(x) = 0 if x <= X_IFACE, and u(x) = (x-X_IFACE)² if x > X_IFACE.
    # u'(x) = 0 if x<=X_IFACE, and 2(x-X_IFACE) if x > X_IFACE. (C¹)
    # u''(x) = 0 if x<=X_IFACE, and 2 if x > X_IFACE.
    # The jump in the second derivative is ⟦u''⟧ = 2 - 0 = 2.
    # The only non-zero component of the Hessian jump is ⟦H_xx⟧ = 2.
    
    def manufactured_sol_pos(x, y):
        return (x - X_IFACE)**2 if x > X_IFACE else 0.0
    def manufactured_sol_neg(x, y):
        return 0.0 if x > X_IFACE else (x - X_IFACE)**2

    u_pos = Function('u_pos', 'u', dh)
    u_pos.set_values_from_function(manufactured_sol_pos)

    u_neg = Function('u_neg', 'u', dh)
    u_neg.set_values_from_function(manufactured_sol_neg)
    jump_u = Jump(u_pos, u_neg)

    penalty_form = hessian_inner(jump_u, jump_u) * dGhost(defined_on=ghost_domain, level_set=level_set, metadata={"q":4})

    # Analytical Calculation:
    # For Q2 interpolation on the cut elements, the local Hessians are constant
    # but not equal to the exact piecewise values. Compute the discrete jump.
    # Assemble the scalar functional
    assembler_hooks={type(penalty_form.integrand):{'name':'penalty_energy'}}
    F = None
    res = assemble_form(Equation(F, penalty_form),  dof_handler=dh, bcs=[], assembler_hooks=assembler_hooks, backend=backend)
    assembled_energy = res['penalty_energy']
    ghost_pos = _mesh.edge_bitset("ghost_pos")
    ghost_neg = _mesh.edge_bitset("ghost_neg")
    ghost_both = _mesh.edge_bitset("ghost_both")
    hxx_pos_cut, hxx_neg_cut = _cut_hxx(_mesh, X_IFACE)
    expected = (
        (2.0 - hxx_neg_cut) ** 2 * _edge_length_sum(_mesh, ghost_pos)
        + (hxx_pos_cut - 2.0) ** 2 * _edge_length_sum(_mesh, ghost_neg)
        + (hxx_pos_cut - hxx_neg_cut) ** 2 * _edge_length_sum(_mesh, ghost_both)
    )
    assert np.isclose(assembled_energy, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("backend", [ "jit", "python"])
def test_hdotn_scalar_spd(setup_quad2, backend):
    _mesh, ls, ghost, dh, comp = setup_quad2
    u_pos = TrialFunction("u", "u_pos", dh, side="+")
    v_pos = TestFunction( "u", "v_pos", dh, side="+")
    u_neg = TrialFunction("u", "u_neg", dh, side="-")
    v_neg = TestFunction( "u", "v_neg", dh, side="-")

    a = inner(hdotn(Jump(u_pos, u_neg)), hdotn(Jump(v_pos, v_neg))) * dGhost(
        defined_on=ghost, level_set=ls, metadata={"q": 6}
    )
    K, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend=backend)
    K = K.toarray()
    assert np.allclose(K, K.T, atol=1e-12)
    eig = np.linalg.eigvalsh(K)
    assert np.all(eig >= -1.5e-10)

@pytest.mark.parametrize("backend", [ "jit", "python"])
def test_hdotn_scalar_zero_for_quadratic(setup_quad2, backend):
    _mesh, ls, ghost, dh, comp = setup_quad2
    uh = Function("u", "u", dh)
    uh.set_values_from_function(lambda x, y: x**2 + y**2)

    Eform = inner(hdotn(Jump(uh)), hdotn(Jump(uh))) * dGhost(
        defined_on=ghost, level_set=ls, metadata={"q": 4}
    )
    hooks = {type(Eform.integrand): {"name": "E"}}
    res = assemble_form(Equation(None, Eform), dof_handler=dh,
                        bcs=[], assembler_hooks=hooks, backend=backend)
    assert abs(res["E"]) < 1e-12

@pytest.mark.parametrize("backend", [ "jit", "python"])
def test_hdotn_scalar_known_value(setup_quad2, backend):
    _mesh, ls, ghost, dh, comp = setup_quad2
    pos_func = lambda x, y: (x - X_IFACE) ** 2 if x > X_IFACE else 0.0
    neg_func = lambda x, y: 0.0 if x > X_IFACE else (x - X_IFACE) ** 2
    u_pos = Function("u_pos", "u", dh)
    u_pos.set_values_from_function(pos_func)
    u_neg = Function("u_neg", "u", dh)
    u_neg.set_values_from_function(neg_func)
    jump_u = Jump(u_pos, u_neg)

    Eform = inner(hdotn(jump_u), hdotn(jump_u)) * dGhost(
        defined_on=ghost, level_set=ls, metadata={"q": 4}
    )
    hooks = {type(Eform.integrand): {"name": "E"}}
    res = assemble_form(Equation(None, Eform), dof_handler=dh,
                        bcs=[], assembler_hooks=hooks, backend=backend)
    ghost_pos = _mesh.edge_bitset("ghost_pos")
    ghost_neg = _mesh.edge_bitset("ghost_neg")
    hxx_pos_cut, hxx_neg_cut = _cut_hxx(_mesh, X_IFACE)
    expected = (
        (2.0 - hxx_neg_cut) ** 2 * _edge_length_weighted(_mesh, ghost_pos, 0)
        + (hxx_pos_cut - 2.0) ** 2 * _edge_length_weighted(_mesh, ghost_neg, 0)
    )
    assert np.isclose(res["E"], expected, rtol=1e-10, atol=1e-12)
