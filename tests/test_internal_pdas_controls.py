import numpy as np
import scipy.sparse as sp

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import NewtonParameters, PdasNewtonSolver, VIParameters
from pycutfem.ufl.expressions import Constant, Function, TestFunction as UflTestFunction, TrialFunction
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad


def _build_scalar_vi_solver(*, vi_params: VIParameters) -> PdasNewtonSolver:
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        poly_order=1,
        element_type="quad",
    )
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    u_k = Function(name="u_k", field_name="u", dof_handler=dh)
    u_n = Function(name="u_n", field_name="u", dof_handler=dh)
    v = UflTestFunction(field_name="u", dof_handler=dh)
    du = TrialFunction(field_name="u", dof_handler=dh)

    residual_form = u_k * v * dx()
    jacobian_form = du * v * dx()
    solver = PdasNewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(newton_tol=1.0e-12, max_newton_iter=5, line_search=True),
        vi_params=vi_params,
        backend="python",
    )
    solver.set_box_bounds(by_field={"u": (0.0, 1.0)})
    return solver


def _build_scalar_nonlinear_vi_solver(
    *,
    vi_params: VIParameters,
    target: float = 0.25,
    initial_guess: float = 0.8,
) -> tuple[PdasNewtonSolver, Function, Function]:
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        poly_order=1,
        element_type="quad",
    )
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    u_k = Function(name="u_k", field_name="u", dof_handler=dh)
    u_n = Function(name="u_n", field_name="u", dof_handler=dh)
    u_k.nodal_values[:] = float(initial_guess)
    u_n.nodal_values[:] = float(initial_guess)

    v = UflTestFunction(field_name="u", dof_handler=dh)
    du = TrialFunction(field_name="u", dof_handler=dh)
    target_c = Constant(float(target))
    two_c = Constant(2.0)

    residual_form = ((u_k * u_k) - target_c) * v * dx()
    jacobian_form = (two_c * u_k * du) * v * dx()

    solver = PdasNewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(newton_tol=1.0e-12, max_newton_iter=12, line_search=True),
        vi_params=vi_params,
        backend="python",
    )
    solver.set_box_bounds(by_field={"u": (0.0, 1.0)})
    return solver, u_k, u_n


def test_internal_pdas_hysteresis_prevents_immediate_release() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(c=2.0, enter_tol=1.0e-3, leave_tol=5.0e-3)
    )
    lo_red, hi_red = solver._bounds_reduced()
    x_red = lo_red.copy()
    c_red = np.full(x_red.shape, 2.0, dtype=float)

    _, act_lo, act_hi = solver._pdas_residual_and_sets(
        x_red, np.full(x_red.shape, 2.0e-3), lo_red, hi_red, c_red=c_red
    )
    assert np.all(act_lo)
    assert not np.any(act_hi)

    _, act_lo, act_hi = solver._pdas_residual_and_sets(
        x_red, np.full(x_red.shape, -2.0e-3), lo_red, hi_red, c_red=c_red
    )
    assert np.all(act_lo)
    assert not np.any(act_hi)

    _, act_lo, act_hi = solver._pdas_residual_and_sets(
        x_red, np.full(x_red.shape, -1.0e-2), lo_red, hi_red, c_red=c_red
    )
    assert not np.any(act_lo)
    assert not np.any(act_hi)


def test_internal_pdas_persistence_requires_repeated_state_change() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(c=1.0, enter_tol=1.0e-4, leave_tol=1.0e-4, active_set_persistence=1)
    )
    lo_red, hi_red = solver._bounds_reduced()
    x_red = lo_red.copy()

    _, act_lo, _ = solver._pdas_residual_and_sets(
        x_red, np.full(x_red.shape, 1.0e-3), lo_red, hi_red
    )
    assert np.all(act_lo)

    _, act_lo, _ = solver._pdas_residual_and_sets(
        x_red, np.full(x_red.shape, -1.0e-3), lo_red, hi_red
    )
    assert np.all(act_lo)

    _, act_lo, _ = solver._pdas_residual_and_sets(
        x_red, np.full(x_red.shape, -1.0e-3), lo_red, hi_red
    )
    assert not np.any(act_lo)


def test_internal_pdas_inactive_regularization_shifts_only_inactive_block() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(inactive_reg_lambda0=0.5, inactive_reg_min_diag=1.0e-12)
    )

    A_base = sp.diags([2.0, 3.0, 4.0], format="csr")
    A_mod = solver._apply_identity_rows(A_base, np.array([0], dtype=int))
    active = np.array([True, False, False], dtype=bool)

    A_reg, add = solver._vi_apply_inactive_regularization(A_mod, A_base, active, 0.5)
    diag = np.asarray(A_reg.diagonal(), dtype=float)

    assert np.allclose(add, np.array([0.0, 1.5, 2.0]))
    assert np.allclose(diag, np.array([1.0, 4.5, 6.0]))


def test_internal_pdas_soft_active_step_keeps_strong_actives_hard() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            active_step_delta_active_trigger=5,
            active_step_soft_alpha=0.25,
        )
    )
    S_red = np.array([1.0, -2.0, 3.0, -4.0], dtype=float)
    active = np.array([True, True, False, True], dtype=bool)
    strong = np.array([True, False, False, False], dtype=bool)

    step, used_soft = solver._vi_apply_active_step_policy(
        alpha=1.0,
        S_red=S_red,
        active_mask=active,
        strong_active_mask=strong,
        active_delta_ref=8,
    )

    assert used_soft
    assert np.allclose(step, np.array([1.0, -0.5, 3.0, -1.0]))


def test_internal_pdas_unconstrained_lm_system_matches_shifted_jacobian() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            unconstrained_lm=True,
            unconstrained_lm_lambda0=0.5,
            unconstrained_lm_min_diag=1.0e-12,
        )
    )
    A = sp.csr_matrix(np.array([[3.0, 1.0], [0.0, 2.0]], dtype=float))
    R = np.array([1.0, -4.0], dtype=float)

    H, rhs, diag = solver._vi_build_unconstrained_lm_system(A, R, 0.5)

    assert np.allclose(diag, np.array([3.0, 2.0]))
    assert np.allclose(H.toarray(), A.toarray() + 0.5 * np.diag(diag))
    assert np.allclose(rhs, -R)


def test_internal_pdas_unconstrained_lm_model_reports_finite_actual_ratio() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            unconstrained_lm=True,
            unconstrained_lm_lambda0=0.25,
            unconstrained_lm_min_diag=1.0e-12,
        )
    )
    A = sp.csr_matrix(np.array([[2.0, 0.0], [0.0, 1.0]], dtype=float))
    R = np.array([2.0, -1.0], dtype=float)
    step = np.array([-0.5, 0.5], dtype=float)
    diag = np.array([2.0, 1.0], dtype=float)
    phi_try = 0.75
    phi0, model_phi, pred_model, rho = solver._vi_unconstrained_lm_model(
        R, A, step, 0.25, diag, phi_try=phi_try
    )

    lin_res = R + A @ step
    expected_phi0 = 0.5 * float(R @ R)
    expected_model_phi = 0.5 * float(lin_res @ lin_res)
    expected_pred_model = expected_phi0 - expected_model_phi
    expected_rho = (expected_phi0 - phi_try) / expected_phi0

    assert np.isclose(phi0, expected_phi0)
    assert np.isclose(model_phi, expected_model_phi)
    assert np.isclose(pred_model, expected_pred_model)
    assert np.isfinite(rho)
    assert np.isclose(rho, expected_rho)


def test_internal_pdas_unconstrained_lm_solves_scalar_constrained_root() -> None:
    solver, u_k, u_n = _build_scalar_nonlinear_vi_solver(
        vi_params=VIParameters(
            c=1.0,
            unconstrained_lm=True,
            unconstrained_lm_lambda0=1.0e-2,
            unconstrained_lm_lambda_max=1.0e2,
            unconstrained_lm_growth=5.0,
            unconstrained_lm_decay=0.5,
            unconstrained_lm_accept_ratio=1.0e-3,
            unconstrained_lm_good_ratio=0.25,
            unconstrained_lm_max_tries=6,
        ),
        target=0.25,
        initial_guess=0.8,
    )

    _, converged, n_iters = solver._newton_loop([u_k], [u_n], {}, [])

    assert converged
    assert n_iters <= 8
    assert np.allclose(np.asarray(u_k.nodal_values, dtype=float), 0.5, atol=1.0e-8)
    assert np.all((np.asarray(u_k.nodal_values, dtype=float) >= 0.0) & (np.asarray(u_k.nodal_values, dtype=float) <= 1.0))
