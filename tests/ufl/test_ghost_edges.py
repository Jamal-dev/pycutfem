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
from pycutfem.core.levelset import LevelSetFunction, CircleLevelSet
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.domain_manager import get_domain_bitset

# --- UFL / compiler ----------------------------------------------------------
from pycutfem.ufl.expressions import Function, TrialFunction, TestFunction, Derivative, Constant, Jump, VectorTrialFunction, VectorTestFunction
from pycutfem.ufl.measures import dGhost, dx
from pycutfem.ufl.compilers import FormCompiler
from tests.ufl.test_face_integrals import dof_handler
from pycutfem.ufl.forms import BoundaryCondition, assemble_form
from pycutfem.io.visualization import plot_mesh_2
import matplotlib.pyplot as plt


# Lx,Ly = 2.0, 1.0
# level_set = CircleLevelSet(radius=0.5, center=(Lx/2, Ly/2))
# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def hessian_inner(u, v):
    if getattr(u, "num_components", 1) == 1:      # scalar
        return _hess_comp(u, v)

    # vector: sum component-wise
    return sum(_hess_comp(u[i], v[i]) for i in range(u.num_components))


def _hess_comp(a, b):
    return (Derivative(a,2,0)*Derivative(b,2,0) +
            2*Derivative(a,1,1)*Derivative(b,1,1) +
            Derivative(a,0,2)*Derivative(b,0,2))

def _grad_comp(a, b):
    return (Derivative(a,1,0) * Derivative(b, 1,0) + Derivative(a, 0, 1) * Derivative(b, 0, 1))

def grad_inner(u, v):
    if getattr(u, "num_components", 1) == 1:  # scalar case
        return _grad_comp(u, v)

    # vector case: sum component-wise
    if u.numcomponents == v.num_components == 2:
        return (
            Derivative(u[0], 1, 0) * Derivative(v[0], 1, 0)+
            Derivative(u[0], 0, 1) * Derivative(v[0], 0, 1) +
            Derivative(u[1], 1, 0) * Derivative(v[1], 1, 0) +
            Derivative(u[1], 0, 1) * Derivative(v[1], 0, 1)
        )
    else : raise ValueError("Unsupported number of components for gradient inner product.")

class VerticalLineLevelSet(LevelSetFunction):
    """ φ(x,y) = x - c   (vertical line x = c) """
    def __init__(self, c: float = 1.0):
        self.c = float(c)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x[..., 0] - self.c          # x - c

    def gradient(self, x: np.ndarray) -> np.ndarray:
        g = np.array([1.0, 0.0])           # ∇φ = (1,0)
        return g if x.ndim == 1 else np.tile(g, (x.shape[0], 1))
class LineLevelSet(LevelSetFunction):
    """
    Zero set is the straight line  y = m * x + b.
    Positive side:  y > m * x + b
    Negative side:  y < m * x + b
    """
    def __init__(self, m: float, b: float = 0.0):
        self.m = float(m)
        self.b = float(b)

    # ---- value ------------------------------------------------------
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x can be shape (..., 2)
        return x[..., 1] - self.m * x[..., 0] - self.b

    # ---- gradient (constant) ---------------------------------------
    def gradient(self, x: np.ndarray) -> np.ndarray:
        g = np.array([-self.m, 1.0])          # ∇φ = (-m, 1)
        # broadcast to match the input
        if x.ndim == 1:
            return g
        return np.tile(g, (x.shape[0], 1))

class AffineLevelSet(LevelSetFunction):
    """
    φ(x, y) = a * x + b * y + c
    Any straight line: choose (a, b, c) so that φ=0 is the line.
    """
    def __init__(self, a: float, b: float, c: float):
        self.a, self.b, self.c = float(a), float(b), float(c)

    # ---- value ------------------------------------------------------
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.array(x)  # Ensure x is a numpy array
        # Works with shape (2,) or (..., 2)
        return self.a * x[..., 0] + self.b * x[..., 1] + self.c

    # ---- gradient ---------------------------------------------------
    def gradient(self, x: np.ndarray) -> np.ndarray:
        g = np.array([self.a, self.b])
        return g if x.ndim == 1 else np.tile(g, (x.shape[0], 1))

    # ---- optional: signed-distance normalisation --------------------
    def normalised(self):
        """Return a copy scaled so that ‖∇φ‖ = 1 (signed-distance)."""
        norm = np.hypot(self.a, self.b)
        return AffineLevelSet(self.a / norm, self.b / norm, self.c / norm)


@pytest.fixture(scope="module")
def setup_quad2():
    """2×1 quadratic mesh cut vertically at x=1."""
    poly_order = 2
    nodes, elements_connectivity, edge_connectivity, corner_nodes = structured_quad(2.0, 1.0, nx=20, ny=5, poly_order=poly_order)
    mesh = Mesh(nodes = nodes,
                element_connectivity = elements_connectivity,
                edges_connectivity = edge_connectivity,
                elements_corner_nodes = corner_nodes,
                element_type="quad",
                poly_order=poly_order)
    
    level_set = AffineLevelSet(a=1.0, b=0, c=-1.0)  # Vertical line at x=1

    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)

    ghost = get_domain_bitset(mesh, "edge", "ghost")
    fig, ax = plt.subplots(figsize=(10, 8))
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

@pytest.mark.parametrize("backend", ["python", "jit"])
def test_hessian_penalty_spd(setup_quad2,backend):
    _mesh, ls, ghost, dh, comp = setup_quad2
    u = TrialFunction(field_name="u",name="u_trial",dof_handler=dh) 
    v = TestFunction( field_name="u",name="v_test",dof_handler=dh)

    a = hessian_inner(Jump(u), Jump(v)) * dGhost(defined_on=ghost, level_set=ls, metadata={"derivs": {(2,0),(1,1),(0,2)}})
    A, _ = assemble_form(a == Constant(0) * dx, dof_handler=dh, bcs=[],backend=backend)
    K = A.toarray()

    # symmetric
    assert np.allclose(K, K.T, atol=1e-12)
    # positive semi‑definite
    evals = np.linalg.eigvalsh(K)
    assert np.all(evals >= -1e-10)

# ---------------------------------------------------------------------------
# 2. Zero‑jump check – quadratic function (constant Hessian)
# ---------------------------------------------------------------------------

def test_hessian_energy_zero_for_quadratic(setup_quad2):
    _mesh, ls, ghost, dh, comp = setup_quad2

    uh = Function(name="uh", field_name="u", dof_handler=dh)
    uh.set_values_from_function(lambda x, y: x**2 + y**2)  # constant Hessian
    jump_u = Jump(uh)
    energy_form = hessian_inner(jump_u, jump_u) * dGhost(defined_on=ghost, level_set=ls, metadata={"q":4})
    F = Constant(0) * dx  # dummy lhs; we need scalar assembly path
    assembler_hooks={type(energy_form.integrand):{'name':'E'}}
    res = assemble_form(F == energy_form,  dof_handler=dh, bcs=[], assembler_hooks=assembler_hooks)
    assert abs(res["E"]) < 1e-12

# ---------------------------------------------------------------------------
# 3. Analytic value – manufactured jump 2 on Hessian xx
# ---------------------------------------------------------------------------

def test_hessian_energy_known_value(setup_quad2):
    _mesh, ls, ghost, dh, comp = setup_quad2

    def piecewise(x, y):
        return np.where(x > 1.0, (x - 1.0) ** 2, 0.0)

    uh = Function(name ="u", field_name="u", dof_handler=dh)
    uh.set_values_from_function(piecewise)

    energy_form = hessian_inner(Jump(uh), Jump(uh)) * dGhost(defined_on=ghost, level_set=ls, metadata={"q":4})
    assembler_hooks={type(energy_form.integrand):{'name':'E'}}
    F = Constant(0) * dx
    res = assemble_form(F == energy_form,  dof_handler=dh, bcs=[], assembler_hooks=assembler_hooks)

    assembled = res["E"]

    expected = 4.0 * 1.0  # jump 2, length 1
    assert np.isclose(assembled, expected, rtol=1e-2)

def test_scalar_jump_penalty_spd(setup_quad2):
    """Ghost penalty K = ∫_Γ ⟦u⟧⟦v⟧ ds must be symmetric PSD."""
    _mesh, level_set, ghost_domain, dh, compiler = setup_quad2

    u = TrialFunction(field_name="u",name="u",dof_handler=dh) 
    v = TestFunction( field_name="u",name="v",dof_handler=dh)
    a = Jump(u) * Jump(v) * dGhost(defined_on=ghost_domain, level_set=level_set, metadata={"q":4})

    K, _ = compiler.assemble(a == Constant(0.0) * dx)  # dummy RHS
    Kd = K.toarray()
    assert np.allclose(Kd, Kd.T), 'K not symmetric'
    eig = np.linalg.eigvalsh(Kd)
    assert np.all(eig >= -1e-12), 'K not PSD'

### Test 4: Mathematical Exactness (Zero-Jump) ###
def test_hessian_penalty_exactness_for_quadratics(setup_quad2):
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
    F = Constant(0) * dx
    res = assemble_form(F == penalty_form,  dof_handler=dh, bcs=[], assembler_hooks=assembler_hooks)
    assembled_energy = res['penalty_energy']

    assert abs(assembled_energy) < 1e-12

### Test 3: Quantitative Correctness (Constant-Jump) ###
def test_hessian_penalty_quantitative_value(setup_quad2):
    """
    Tests that the assembled value of the Hessian penalty for a manufactured
    solution matches the known analytical value.
    """
    _mesh, level_set, ghost_domain, dh, compiler = setup_quad2

    # We use a function u that is C¹ continuous, but whose second derivative jumps.
    # u(x) = 0 if x <= 1, and u(x) = (x-1)² if x > 1.
    # u'(x) = 0 if x<=1, and 2(x-1) if x > 1. (u'(1)=0, so C¹)
    # u''(x) = 0 if x<=1, and 2 if x > 1.
    # The jump in the second derivative is ⟦u''⟧ = 2 - 0 = 2.
    # The only non-zero component of the Hessian jump is ⟦H_xx⟧ = 2.
    
    def manufactured_sol(x, y):
        return (x - 1.0)**2 if x > 1.0 else 0.0

    u_h = Function('u', 'u', dh)
    u_h.set_values_from_function(manufactured_sol)

    penalty_form = hessian_inner(Jump(u_h), Jump(u_h)) * dGhost(defined_on=ghost_domain, level_set=level_set, metadata={"q":4})

    # Analytical Calculation:
    # The integral is ∫ |⟦H(u)⟧|² ds ≈ ∫ (⟦∂²u/∂x²⟧)² ds = ∫ (2)² ds = 4 * length.
    # The ghost edge runs from (1,0) to (1,1), so its length is 1.
    expected_energy = 4.0 * 1.0

    # Assemble the scalar functional
    assembler_hooks={type(penalty_form.integrand):{'name':'penalty_energy'}}
    F = Constant(0) * dx
    res = assemble_form(F == penalty_form,  dof_handler=dh, bcs=[], assembler_hooks=assembler_hooks)
    assembled_energy = res['penalty_energy']
    assert np.isclose(assembled_energy, expected_energy, rtol=1e-2), \
        f"Expected {expected_energy}, got {assembled_energy} for the Hessian penalty energy."

# E. Vector jump of gradients
# ---------------------------

# def test_grad_jump_energy_vanishes_for_linear(setup_quad2):
#     """For linear **vector** field, grad jump is constant ⇒ penalty non‑zero; but linear *continuous* field should give zero."""
#     mesh, level_set, ghost_domain, dh, compiler = setup_quad2

#     # Build vector function space: two components, each Q2
#     vme = MixedElement(mesh, {'U': 2, 'V': 2})
#     vdh = DofHandler(vme)
#     vcompiler = FormCompiler(vdh, quadrature_order=4)

#     U, V = VectorTrialFunction('UV', dim=2), VectorTestFunction('UV', dim=2)

#     a = Jump(grad(U)) @ Jump(grad(V)) * dGhost(defined_on=ghost_domain, level_set=level_set)

#     # Continuous linear vector field – same on both sides ⇒ grad jump = 0
#     uh = Function('UV', 'UV', vdh)
#     uh.set_values_from_function(lambda x, y: np.array([x, y]))

#     # Assemble energy
#     _, F = vcompiler.assemble(VectorTestFunction('UV')*dGhost == a,
#                               assembler_hooks={Function: {'name': 'penalty_energy'}})
#     E = vcompiler.ctx['scalar_results']['penalty_energy']
#     assert abs(E) < 1e-12, 'Pen. energy should vanish for continuous linear field'
