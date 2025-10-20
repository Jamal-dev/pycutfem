
import numpy as np
import pytest

# --- Core / mesh & FE plumbing ---
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.levelset import AffineLevelSet

# --- UFL-like layer ---
from pycutfem.ufl.expressions import TrialFunction, TestFunction, Constant
from pycutfem.ufl.measures import dx
from pycutfem.ufl.forms import assemble_form
from pycutfem.ufl.expressions import Function, inner, grad
from pycutfem.solvers.nonlinear_solver import NewtonSolver, TimeStepperParameters


# ---------------------------
# Helpers
# ---------------------------

def assemble_rhs_area(mesh, level_set=None, side='+', q=6, poly_order=1, backend='jit'):
    """
    Assemble the linear functional F_i = ∫_{Ω_phys} 1 * φ_i dx and return sum(F).
    Because Lagrange bases form a partition of unity (∑_i φ_i == 1), sum(F) == |Ω_phys|.
    If `level_set` is None, integrates over the full mesh.
    If `level_set` is provided, integrates over the chosen side (phi>0 if side='+', else phi<0).
    """
    me = MixedElement(mesh, field_specs={'u': poly_order})
    dh = DofHandler(me, method='cg')

    u = TrialFunction('u', dof_handler=dh)
    v = TestFunction('u', dof_handler=dh)

    meas = dx() if level_set is None else dx(level_set=level_set, metadata={'side': side})

    # zero bilinear form to keep assembler API consistent
    a = (Constant(0.0) * u * v) * dx()
    f = (Constant(1.0) * v) * meas

    K, F = assemble_form(a == f, dof_handler=dh, bcs=[], quad_order=q, backend=backend)
    return float(F.sum())

def assemble_mass_matrix(mesh, level_set=None, side='+', q=6, poly_order=1, backend='jit'):
    """Assemble the scalar mass matrix M_ij = ∫ φ_i φ_j dx on the requested domain."""
    me = MixedElement(mesh, field_specs={'u': poly_order})
    dh = DofHandler(me, method='cg')

    u = TrialFunction('u', dof_handler=dh)
    v = TestFunction('u', dof_handler=dh)

    meas = dx() if level_set is None else dx(level_set=level_set, metadata={'side': side})

    a = (u * v) * meas
    K, F = assemble_form(a == (Constant(0.0) * v) * dx(), dof_handler=dh, bcs=[], quad_order=q, backend=backend)
    return K


# ---------------------------
# Tests
# ---------------------------
@pytest.mark.parametrize("backend", ['python', 'jit'])
def test_area_partition_single_cell_linear_cut(backend):
    """On a single unit quad, phi(x,y)=x-alpha ⇒ |Ω_{phi>0}| = 1 - alpha; |Ω_{phi<0}| = alpha."""
    # Mesh: one 1x1 quad
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(nodes, element_connectivity=elems, elements_corner_nodes=corners, poly_order=1, element_type='quad')

    def make_phi(alpha):
        # φ(x,y) = x - alpha  →  a=1, b=0, c=-alpha
        return AffineLevelSet(1.0, 0.0, -alpha)

    for alpha in [0.1, 0.25, 0.5, 0.9]:
        phi = make_phi(alpha)
        area_plus  = assemble_rhs_area(mesh, level_set=phi, side='+', q=6, poly_order=1, backend=backend)
        area_minus = assemble_rhs_area(mesh, level_set=phi, side='-', q=6, poly_order=1, backend=backend)
        assert np.isclose(area_plus,  1.0 - alpha, atol=5e-4, rtol=1e-4)
        assert np.isclose(area_minus, alpha,       atol=5e-4, rtol=1e-4)
        assert np.isclose(area_plus + area_minus, 1.0, atol=5e-4, rtol=1e-4)

@pytest.mark.parametrize("backend", ['python', 'jit'])
def test_mass_matrix_partition_equals_full(backend):
    """M^{+} + M^{-} == M^{full} on a single cut element (conservation by partition)."""
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(nodes, element_connectivity=elems, elements_corner_nodes=corners, poly_order=1, element_type='quad')

    phi = AffineLevelSet(1.0, 0.0, -0.37)

    M_plus  = assemble_mass_matrix(mesh, level_set=phi, side='+', q=6, poly_order=1, backend=backend)
    M_minus = assemble_mass_matrix(mesh, level_set=phi, side='-', q=6, poly_order=1, backend=backend)
    M_full  = assemble_mass_matrix(mesh, level_set=None,       side='+', q=6, poly_order=1, backend=backend)

    diff = (M_plus + M_minus - M_full).toarray()
    assert np.allclose(diff, 0.0, atol=5e-6, rtol=1e-6)


@pytest.mark.parametrize("backend", ['python', 'jit'])
def test_levelset_matches_fitted_when_interface_aligned(backend):
    """When phi=0 aligns with an element boundary, level-set volume equals fitted volume."""
    # 2x1 mesh on [0,1]x[0,1]; interface at x=0.5 lies on the inter-element edge
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=1)
    mesh = Mesh(nodes, element_connectivity=elems, elements_corner_nodes=corners, poly_order=1, element_type='quad')

    phi = AffineLevelSet(1.0, 0.0, -0.5)

    # Using level-set measure (phi>0)
    area_ls = assemble_rhs_area(mesh, level_set=phi, side='+', q=6, poly_order=1, backend=backend)

    # Fitted area: sum areas of elements with centroid x > 0.5 (right half)
    right_ids = [eid for eid in range(len(mesh.elements_list))
                 if mesh.elements_list[eid].centroid()[0] > 0.5]
    fitted_area = sum(mesh.areas()[eid] for eid in right_ids)

    assert np.isclose(area_ls, fitted_area, atol=5e-6, rtol=1e-6)
    assert np.isclose(area_ls, 0.5, atol=5e-6, rtol=1e-6)


# ----------------------------------------------------------------------
# Newton-solver tests for cut-volume assembly (JIT backend)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("side", ['+', '-'])
def test_newton_mass_projection_on_cut_domain(side):
    """
    Linear mass problem on a single cut cell:
        Find u such that  ∫_{Ω_side} u v dx = ∫_{Ω_side} 1·v dx.
    Exact solution in CG space is u ≡ 1, so Newton converges in one step.
    This validates the cut-volume path in compile_multi used by Newton.
    """
    # Single element mesh on [0,1]x[0,1]
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(nodes, element_connectivity=elems, elements_corner_nodes=corners,
                poly_order=1, element_type='quad')

    # Linear cut: phi(x,y) = x - alpha (cuts the single quad)
    alpha = 0.37
    phi = AffineLevelSet(1.0, 0.0, -alpha)

    me = MixedElement(mesh, field_specs={'u': 1})
    dh = DofHandler(me, method='cg')

    # Unknown and test/trial symbols
    u_k  = Function(name= 'u_k', field_name='u', dof_handler=dh)
    v  = TestFunction(field_name='u', dof_handler=dh)
    u = TrialFunction(field_name='u', dof_handler=dh)

    # Measure restricted to the chosen side of the cut
    meas = dx(level_set=phi, metadata={'side': side})

    # Residual and Jacobian forms (linear problem)
    residual_form = (u_k * v - Constant(1.0) * v) * meas
    jacobian_form = (u * v) * meas

    # Newton driver (JIT backend; no BCs needed for pure mass on a single element)
    solver = NewtonSolver(
        residual_form, jacobian_form,
        dof_handler=dh, mixed_element=me,
        bcs=[], bcs_homog=[],
        backend='jit',
    )

    # Initial/previous states
    u_prev = Function(name='u_n', field_name='u', dof_handler=dh)
    u_prev.nodal_values[:] = 0.0
    u_k.nodal_values[:] = 0.0

    # One pseudo-time step is enough for a linear system
    delta, nsteps, _elapsed = solver.solve_time_interval(
        functions=[u_k], prev_functions=[u_prev],
        time_params=TimeStepperParameters(dt=1.0, max_steps=1, stop_on_steady=True),
    )

    # Expect the nodal solution to be (close to) 1.0 everywhere
    assert np.allclose(u_k.nodal_values, 1.0, atol=1e-10, rtol=0.0)

    # Residual at the solution should be ~ 0
    # (assemble residual only, no need for the Jacobian here)
    R_only = solver._assemble_system({'u_k': u_k, 'u_n': u_prev}, need_matrix=False)[1]
    assert np.linalg.norm(R_only, np.inf) < 1e-11


@pytest.mark.parametrize("side", ['+', '-'])
def test_newton_helmholtz_on_cut_domain(side):
    """
    Helmholtz-type linear problem on a cut cell:
        ∫_{Ω_side} (∇u·∇v + u v) dx = ∫_{Ω_side} 1·v dx.
    This touches the gradient tables in the cut-volume precompute.
    We assert that Newton drives the residual below a tight tolerance.
    """
    # Mesh & cut as above
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(nodes, element_connectivity=elems, elements_corner_nodes=corners,
                poly_order=1, element_type='quad')
    phi = AffineLevelSet(1.0, 0.0, -0.41)

    me = MixedElement(mesh, field_specs={'u': 1})
    dh = DofHandler(me, method='cg')

    u_k  = Function(name='u_k',field_name='u', dof_handler=dh)
    v  = TestFunction(field_name='u', dof_handler=dh)
    u = TrialFunction(field_name='u', dof_handler=dh)

    meas = dx(level_set=phi, metadata={'side': side})

    # Residual/Jacobian
    residual_form = (inner(grad(u_k), grad(v)) + u_k * v - Constant(1.0) * v) * meas
    jacobian_form = (inner(grad(u), grad(v)) + u * v) * meas

    solver = NewtonSolver(
        residual_form, jacobian_form,
        dof_handler=dh, mixed_element=me,
        bcs=[], bcs_homog=[],
        backend='jit',
    )

    # Start from zero; a few Newton iterations suffice (linear system)
    u_prev = Function(name='u_n', field_name='u', dof_handler=dh)
    u_prev.nodal_values[:] = 0.0
    u_k.nodal_values[:] = 0.0

    delta, nsteps, _elapsed = solver.solve_time_interval(
        functions=[u_k], prev_functions=[u_prev],
        time_params=TimeStepperParameters(dt=1.0, max_steps=1, stop_on_steady=True),
    )

    # Check the assembled residual at the solution
    _, R = solver._assemble_system({'u_k': u_k, 'u_n': u_prev}, need_matrix=False)
    assert np.linalg.norm(R, np.inf) < 1e-9