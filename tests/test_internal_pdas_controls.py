import numpy as np
import scipy.sparse as sp
import pytest

from examples.biofilms.benchmarks.seboldt.paper1_benchmark7_seboldt import (
    _build_bcs,
    _build_forms,
    _create_problem,
    _latent_forward_array,
    _latent_inverse_array,
)
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import (
    LinearEqualityConstraint,
    NewtonParameters,
    NewtonSolver,
    PdasNewtonSolver,
    VIParameters,
    _PreparedLinearEqualities,
    _vi_filter_abs_floor,
    _vi_filter_threshold,
)
from pycutfem.ufl.expressions import Constant, Function, TestFunction as UflTestFunction, TrialFunction
from pycutfem.ufl.forms import Equation, assemble_form
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


def _build_two_field_vi_solver(*, vi_params: VIParameters):
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        poly_order=1,
        element_type="quad",
    )
    me = MixedElement(mesh, field_specs={"alpha": 1, "phi": 1})
    dh = DofHandler(me, method="cg")

    alpha_k = Function(name="alpha_k", field_name="alpha", dof_handler=dh)
    alpha_n = Function(name="alpha_n", field_name="alpha", dof_handler=dh)
    phi_k = Function(name="phi_k", field_name="phi", dof_handler=dh)
    phi_n = Function(name="phi_n", field_name="phi", dof_handler=dh)

    test_alpha = UflTestFunction(field_name="alpha", dof_handler=dh)
    trial_alpha = TrialFunction(field_name="alpha", dof_handler=dh)
    test_phi = UflTestFunction(field_name="phi", dof_handler=dh)
    trial_phi = TrialFunction(field_name="phi", dof_handler=dh)

    residual_form = alpha_k * test_alpha * dx() + phi_k * test_phi * dx()
    jacobian_form = trial_alpha * test_alpha * dx() + trial_phi * test_phi * dx()
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
    solver.set_box_bounds(by_field={"alpha": (0.0, 1.0), "phi": (0.0, 1.0)})
    return solver, dh, alpha_k, alpha_n, phi_k, phi_n


def _build_two_field_newton_solver():
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        poly_order=1,
        element_type="quad",
    )
    me = MixedElement(mesh, field_specs={"alpha": 1, "phi": 1})
    dh = DofHandler(me, method="cg")

    alpha_k = Function(name="alpha_k", field_name="alpha", dof_handler=dh)
    alpha_n = Function(name="alpha_n", field_name="alpha", dof_handler=dh)
    phi_k = Function(name="phi_k", field_name="phi", dof_handler=dh)
    phi_n = Function(name="phi_n", field_name="phi", dof_handler=dh)

    test_alpha = UflTestFunction(field_name="alpha", dof_handler=dh)
    trial_alpha = TrialFunction(field_name="alpha", dof_handler=dh)
    test_phi = UflTestFunction(field_name="phi", dof_handler=dh)
    trial_phi = TrialFunction(field_name="phi", dof_handler=dh)

    residual_form = alpha_k * test_alpha * dx() + phi_k * test_phi * dx()
    jacobian_form = trial_alpha * test_alpha * dx() + trial_phi * test_phi * dx()
    solver = NewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(newton_tol=1.0e-12, max_newton_iter=5, line_search=True),
        backend="python",
    )
    return solver, dh, alpha_k, alpha_n, phi_k, phi_n


def test_vi_filter_threshold_uses_projection_scale_for_zero_baseline() -> None:
    vi_params = VIParameters()
    abs_floor = _vi_filter_abs_floor(vi_params)
    assert abs_floor == pytest.approx(1.0e-12)
    assert _vi_filter_threshold(base=0.0, growth=1.25, abs_floor=abs_floor) == pytest.approx(1.25e-12)


def test_vi_filter_abs_floor_respects_larger_projection_tolerance() -> None:
    vi_params = VIParameters()
    setattr(vi_params, "projection_tol", 5.0e-10)
    assert _vi_filter_abs_floor(vi_params) == pytest.approx(5.0e-10)


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


def test_newton_solver_set_active_fields_restricts_to_requested_field() -> None:
    solver, dh, alpha_k, alpha_n, phi_k, phi_n = _build_two_field_newton_solver()

    active = solver.set_active_fields(["alpha"])
    alpha_slice = np.asarray(dh.get_field_slice("alpha"), dtype=int)

    assert np.array_equal(np.asarray(active, dtype=int), alpha_slice)
    assert np.array_equal(np.asarray(solver.red_to_full, dtype=int), alpha_slice)

    coeffs = {
        alpha_k.name: alpha_k,
        alpha_n.name: alpha_n,
        phi_k.name: phi_k,
        phi_n.name: phi_n,
    }
    A_red, R_red = solver._assemble_system_reduced(coeffs, need_matrix=True)

    assert A_red.shape == (alpha_slice.size, alpha_slice.size)
    assert R_red.shape == (alpha_slice.size,)


def test_newton_logistic_transform_keeps_bounded_fields_in_open_interval() -> None:
    solver, dh, alpha_k, alpha_n, phi_k, phi_n = _build_two_field_newton_solver()
    eps = 1.0e-6
    solver.set_logistic_transform_fields(["alpha", "phi"], eps=eps)

    alpha_k.nodal_values[:] = 0.0
    phi_k.nodal_values[:] = 1.0
    funcs = [alpha_k, phi_k]

    solver._prepare_solver_coordinate_iterate(funcs, [])
    assert np.all(alpha_k.nodal_values > eps * 0.99)
    assert np.all(alpha_k.nodal_values < 1.0 - eps * 0.99)
    assert np.all(phi_k.nodal_values > eps * 0.99)
    assert np.all(phi_k.nodal_values < 1.0 - eps * 0.99)

    step_red = np.zeros((int(len(solver.active_dofs)),), dtype=float)
    alpha_slice = np.asarray(dh.get_field_slice("alpha"), dtype=int).ravel()
    phi_slice = np.asarray(dh.get_field_slice("phi"), dtype=int).ravel()
    step_red[alpha_slice] = 50.0
    step_red[phi_slice] = -50.0
    solver._apply_solver_coordinate_step(step_red, funcs, [])

    assert np.all(alpha_k.nodal_values > eps * 0.99)
    assert np.all(alpha_k.nodal_values < 1.0 - eps * 0.99)
    assert np.all(phi_k.nodal_values > eps * 0.99)
    assert np.all(phi_k.nodal_values < 1.0 - eps * 0.99)


def test_newton_logistic_transform_applies_chain_rule_column_scaling() -> None:
    solver, dh, alpha_k, alpha_n, phi_k, phi_n = _build_two_field_newton_solver()
    solver.set_logistic_transform_fields(["alpha", "phi"], eps=1.0e-8)

    alpha_k.nodal_values[:] = 0.25
    phi_k.nodal_values[:] = 0.75
    funcs = [alpha_k, phi_k]
    coeffs = {
        alpha_k.name: alpha_k,
        alpha_n.name: alpha_n,
        phi_k.name: phi_k,
        phi_n.name: phi_n,
    }
    A_red, _ = solver._assemble_system_reduced(coeffs, need_matrix=True)
    A_step, scale = solver._solver_coordinate_linear_system(A_red, funcs)

    alpha_slice = np.asarray(dh.get_field_slice("alpha"), dtype=int).ravel()
    phi_slice = np.asarray(dh.get_field_slice("phi"), dtype=int).ravel()
    expected_scale = 0.25 * 0.75
    assert np.allclose(scale[alpha_slice], expected_scale)
    assert np.allclose(scale[phi_slice], expected_scale)
    assert np.allclose(A_step.toarray(), A_red.toarray() @ np.diag(scale))


def test_latent_bounded_logit_sigmoid_round_trip_clips_to_open_interval() -> None:
    vals = np.asarray([0.0, 1.0e-8, 0.25, 0.75, 1.0 - 1.0e-8, 1.0], dtype=float)
    eps = 1.0e-6
    latent = _latent_inverse_array(vals, eps=eps, map_kind="sigmoid")
    recovered = _latent_forward_array(latent, map_kind="sigmoid")

    assert np.all(recovered > eps * 0.99)
    assert np.all(recovered < 1.0 - eps * 0.99)
    assert recovered[2] == pytest.approx(0.25, rel=0.0, abs=1.0e-12)
    assert recovered[3] == pytest.approx(0.75, rel=0.0, abs=1.0e-12)


def test_latent_bounded_bcs_duplicate_dirichlet_data_in_logit_coordinates() -> None:
    bcs = _build_bcs(
        fluid_space="cg",
        enable_phi_evolution=False,
        y_interface=0.5,
        eps_alpha=0.05,
        v_in=1.0,
        t_ramp=0.0,
        alpha_bc_mode="equilibrium",
        solid_bc_mode="base_only",
        latent_bounded_fields=("alpha",),
        latent_bounded_eps=1.0e-6,
    )

    alpha_top = [bc for bc in bcs if bc.field == "alpha" and bc.method == "dirichlet" and bc.domain_tag == "top"]
    alpha_latent_top = [
        bc for bc in bcs if bc.field == "alpha_latent" and bc.method == "dirichlet" and bc.domain_tag == "top"
    ]

    assert len(alpha_top) == 1
    assert len(alpha_latent_top) == 1

    alpha_val = float(alpha_top[0].value(0.3, 1.0, 0.0))
    latent_val = float(alpha_latent_top[0].value(0.3, 1.0, 0.0))
    expected = float(
        _latent_inverse_array(np.asarray([alpha_val], dtype=float), eps=1.0e-6, map_kind="sigmoid")[0]
    )

    assert latent_val == pytest.approx(expected, rel=0.0, abs=1.0e-12)


def test_latent_bounded_transport_full_jacobian_fd_consistency() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=1,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=False,
        latent_bounded_transport=True,
    )
    problem["latent_bounded_fields"] = ("alpha",)

    problem["v_k"].set_values_from_function(lambda x, y: np.array([0.18 + 0.04 * x - 0.03 * y, -0.06 + 0.02 * x + 0.015 * y]))
    problem["v_n"].set_values_from_function(lambda x, y: np.array([0.11 + 0.03 * x - 0.02 * y, -0.03 + 0.015 * x + 0.010 * y]))
    problem["vS_k"].set_values_from_function(lambda x, y: np.array([0.05 + 0.02 * x - 0.01 * y, -0.03 + 0.01 * x + 0.015 * y]))
    problem["vS_n"].set_values_from_function(lambda x, y: np.array([0.03 + 0.01 * x - 0.008 * y, -0.015 + 0.008 * x + 0.010 * y]))
    problem["u_k"].set_values_from_function(lambda x, y: np.array([0.015 + 0.010 * x * (1.0 - x), -0.008 + 0.006 * y * (1.0 - y / 1.5)]))
    problem["u_n"].set_values_from_function(lambda x, y: np.array([0.010 + 0.006 * x * (1.0 - x), -0.005 + 0.004 * y * (1.0 - y / 1.5)]))
    problem["p_k"].set_values_from_function(lambda x, y: 0.25 + 0.10 * x - 0.05 * y)
    problem["p_n"].set_values_from_function(lambda x, y: 0.12 + 0.04 * x - 0.03 * y)
    problem["alpha_k"].set_values_from_function(lambda x, y: 0.58 + 0.015 * x + 0.010 * y)
    problem["alpha_n"].set_values_from_function(lambda x, y: 0.54 + 0.010 * x + 0.008 * y)
    problem["mu_k"].set_values_from_function(lambda x, y: 0.10 + 0.020 * x - 0.015 * y)
    problem["mu_n"].set_values_from_function(lambda x, y: 0.07 + 0.015 * x - 0.010 * y)
    problem["alpha_latent_k"].nodal_values[:] = _latent_inverse_array(
        problem["alpha_k"].nodal_values,
        eps=1.0e-8,
        map_kind="sigmoid",
    )
    problem["alpha_latent_n"].nodal_values[:] = _latent_inverse_array(
        problem["alpha_n"].nodal_values,
        eps=1.0e-8,
        map_kind="sigmoid",
    )

    qdeg = 4
    forms = _build_forms(
        problem,
        qdeg=qdeg,
        dt_c=Constant(0.01),
        theta=1.0,
        rho_f=1.0,
        mu_f=0.035,
        mu_b=0.035,
        mu_b_model="mu",
        kappa_inv=1.0e5,
        mu_s=1.67785e5,
        lambda_s=8.22148e6,
        phi_b=0.18,
        M_alpha=0.0,
        gamma_alpha=1.0,
        eps_alpha=0.05,
        solid_visco_eta=0.0,
        gamma_div=0.0,
        gamma_u=0.0,
        u_extension_mode="l2",
        gamma_u_pin=0.0,
        u_cip=0.0,
        u_cip_weight="fluid",
        vS_cip=0.0,
        gamma_vS=None,
        vS_extension_mode=None,
        gamma_vS_pin=None,
        D_phi=0.0,
        phi_diffusion_weight="fluid",
        gamma_phi=5.0,
        phi_supg=0.0,
        phi_cip=0.0,
        alpha_supg=0.0,
        alpha_cip=0.0,
        v_supg=0.0,
        v_supg_mode="streamline",
        v_supg_c_nu=4.0,
        u_supg=0.0,
        v_cip=0.0,
        alpha_regularization="none",
        alpha_reg_gamma=1.0,
        alpha_reg_eps_normal=0.05,
        alpha_reg_eps_tangent=0.0125,
        alpha_reg_eta=1.0e-12,
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        support_physics="internal_conversion",
        skeleton_pressure_mode="whole_domain",
        alpha_biot=None,
        g_t_k=None,
        g_t_n=None,
        traction_weight_k=None,
        traction_weight_n=None,
        solid_model="linear",
        kappa_inv_model="refmap",
        enable_phi_evolution=False,
        include_skeleton_acceleration=True,
        rho_s0_tilde=1.1,
        skeleton_inertia_convection="full",
    )

    dh = problem["dh"]
    jac, res0 = assemble_form(
        Equation(forms.jacobian_form, forms.residual_form),
        dof_handler=dh,
        bcs=[],
        quad_order=qdeg,
        backend="python",
    )
    jac = jac.tocsr()
    res0 = np.asarray(res0, dtype=float)
    field_to_func = {
        "v_x": problem["v_k"].components[0],
        "v_y": problem["v_k"].components[1],
        "p": problem["p_k"],
        "vS_x": problem["vS_k"].components[0],
        "vS_y": problem["vS_k"].components[1],
        "u_x": problem["u_k"].components[0],
        "u_y": problem["u_k"].components[1],
        "alpha": problem["alpha_k"],
        "mu_alpha": problem["mu_k"],
        "alpha_latent": problem["alpha_latent_k"],
    }
    rng = np.random.default_rng(1234)
    probe_sets = {
        "alpha_latent_only": ["alpha_latent"],
        "alpha_pair": ["alpha", "alpha_latent"],
        "random_all": ["v_x", "v_y", "p", "vS_x", "vS_y", "u_x", "u_y", "alpha", "mu_alpha", "alpha_latent"],
    }

    for name, fields in probe_sets.items():
        z = np.zeros(jac.shape[1], dtype=float)
        for fld in fields:
            sl = np.asarray(dh.get_field_slice(fld), dtype=int)
            if sl.size:
                z[sl] = rng.standard_normal(sl.size)
        z /= float(np.linalg.norm(z, ord=np.inf))

        eps = 1.0e-8
        touched: list[tuple[object, np.ndarray, np.ndarray]] = []
        for fld, func in field_to_func.items():
            sl = np.asarray(dh.get_field_slice(fld), dtype=int)
            if sl.size == 0:
                continue
            dz = z[sl]
            if np.allclose(dz, 0.0):
                continue
            old = np.asarray(func.get_nodal_values(sl), dtype=float).copy()
            func.set_nodal_values(sl, old + eps * dz)
            touched.append((func, sl, old))

        _, res1 = assemble_form(
            Equation(None, forms.residual_form),
            dof_handler=dh,
            bcs=[],
            quad_order=qdeg,
            backend="python",
        )
        res1 = np.asarray(res1, dtype=float)
        for func, sl, old in touched:
            func.set_nodal_values(sl, old)

        fd = (res1 - res0) / eps
        lin = jac @ z
        denom = max(1.0, float(np.linalg.norm(fd, ord=np.inf)), float(np.linalg.norm(lin, ord=np.inf)))
        rel = float(np.linalg.norm(fd - lin, ord=np.inf)) / denom
        tol = 1.0e-5 if name == "alpha_pair" else 1.0e-8
        assert rel < tol, f"{name} latent FD mismatch too large: {rel:.3e}"


def test_internal_pdas_set_active_fields_refreshes_reduced_vi_metadata() -> None:
    solver, dh, *_ = _build_two_field_vi_solver(
        vi_params=VIParameters(c=1.0, enter_tol=1.0e-6, leave_tol=1.0e-6)
    )

    solver.set_active_fields(["alpha"])
    lo_red, hi_red = solver._bounds_reduced()
    alpha_slice = np.asarray(dh.get_field_slice("alpha"), dtype=int)

    assert lo_red.shape == (alpha_slice.size,)
    assert hi_red.shape == (alpha_slice.size,)
    assert np.all(lo_red == 0.0)
    assert np.all(hi_red == 1.0)
    assert np.all(np.asarray(solver._vi_red_field_names, dtype=object) == "alpha")


def test_internal_pdas_single_alpha_equality_projection_uses_field_local_shift() -> None:
    solver, dh, alpha_k, alpha_n, phi_k, phi_n = _build_two_field_vi_solver(
        vi_params=VIParameters(c=1.0)
    )
    alpha_k.nodal_values[:] = 0.9
    alpha_n.nodal_values[:] = 0.9
    phi_k.nodal_values[:] = 0.2
    phi_n.nodal_values[:] = 0.2

    alpha_slice = np.asarray(dh.get_field_slice("alpha"), dtype=int).ravel()
    weights_full = np.zeros(int(dh.total_dofs), dtype=float)
    weights_full[alpha_slice] = 1.0
    solver.set_linear_equalities(
        [
            LinearEqualityConstraint(
                name="alpha_mass",
                weights_full=weights_full,
                target=2.0,
                field_name="alpha",
                project_feasible=True,
            )
        ]
    )
    eq_data = solver._vi_prepare_linear_equalities(
        [alpha_k, phi_k],
        [alpha_n, phi_n],
        {},
        [],
    )
    lo_red, hi_red = solver._bounds_reduced()
    x_red = solver._pack_reduced_iterate([alpha_k, phi_k])
    x_proj = solver._project_box_with_prepared_equalities(
        x=x_red,
        lo=lo_red,
        hi=hi_red,
        eq_data=eq_data,
        tol=1.0e-12,
    )

    alpha_mask = np.asarray(solver._vi_red_field_names, dtype=object) == "alpha"
    phi_mask = np.asarray(solver._vi_red_field_names, dtype=object) == "phi"

    assert np.allclose(x_proj[alpha_mask], 0.5)
    assert np.allclose(x_proj[phi_mask], 0.2)


def test_internal_pdas_row_scaling_balances_fieldwise_row_medians() -> None:
    solver, dh, *_ = _build_two_field_vi_solver(vi_params=VIParameters(equation_row_scaling=True))
    alpha_slice = np.asarray(dh.get_field_slice("alpha"), dtype=int)
    phi_slice = np.asarray(dh.get_field_slice("phi"), dtype=int)
    n = int(dh.total_dofs)
    diag = np.ones((n,), dtype=float)
    diag[alpha_slice] = 1.0e1
    diag[phi_slice] = 1.0e-1
    A_red = sp.diags(diag, format="csr")

    row_scale = solver._vi_row_scale_vector(A_red)
    row_abs_scaled = solver._vi_scale_vector_rows(np.asarray(np.abs(A_red).sum(axis=1)).ravel(), row_scale)

    alpha_med = float(np.median(row_abs_scaled[alpha_slice]))
    phi_med = float(np.median(row_abs_scaled[phi_slice]))
    assert alpha_med == pytest.approx(phi_med, rel=1.0e-12, abs=1.0e-12)


def test_internal_pdas_row_scaling_preserves_active_set_with_inverse_c_scaling() -> None:
    solver = _build_scalar_vi_solver(vi_params=VIParameters(c=2.0, equation_row_scaling=True))
    lo_red, hi_red = solver._bounds_reduced()
    x_red = lo_red.copy()
    R_raw = np.full(x_red.shape, 3.0e-3, dtype=float)
    c_raw = np.full(x_red.shape, 2.0, dtype=float)
    row_scale = np.full(x_red.shape, 7.5, dtype=float)
    R_scaled = solver._vi_scale_vector_rows(R_raw, row_scale)
    c_scaled = solver._vi_scale_c_vector(c_raw, row_scale)

    _, act_lo_raw, act_hi_raw = solver._pdas_residual_and_sets(x_red, R_raw, lo_red, hi_red, c_red=c_raw)
    _, act_lo_scaled, act_hi_scaled = solver._pdas_residual_and_sets(
        x_red, R_scaled, lo_red, hi_red, c_red=c_scaled
    )

    assert np.array_equal(act_lo_raw, act_lo_scaled)
    assert np.array_equal(act_hi_raw, act_hi_scaled)


def test_internal_pdas_column_scaling_balances_selected_fieldwise_column_medians() -> None:
    solver, dh, *_ = _build_two_field_vi_solver(
        vi_params=VIParameters(
            variable_column_scaling=True,
            variable_column_scaling_fields=("alpha", "phi"),
        )
    )
    alpha_slice = np.asarray(dh.get_field_slice("alpha"), dtype=int)
    phi_slice = np.asarray(dh.get_field_slice("phi"), dtype=int)
    n = int(dh.total_dofs)
    diag = np.ones((n,), dtype=float)
    diag[alpha_slice] = 1.0e1
    diag[phi_slice] = 1.0e-1
    A_red = sp.diags(diag, format="csr")

    col_scale = solver._vi_col_scale_vector(A_red)
    col_abs_scaled = np.asarray(np.abs(solver._vi_scale_matrix_cols(A_red, col_scale)).sum(axis=0)).ravel()

    alpha_med = float(np.median(col_abs_scaled[alpha_slice]))
    phi_med = float(np.median(col_abs_scaled[phi_slice]))
    assert alpha_med == pytest.approx(phi_med, rel=1.0e-12, abs=1.0e-12)


def test_internal_pdas_stationarity_residual_scales_equality_transpose_term() -> None:
    solver = _build_scalar_vi_solver(vi_params=VIParameters(equation_row_scaling=True))
    R_red = np.array([1.0, -2.0], dtype=float)
    eq_data = _PreparedLinearEqualities(
        names=("mass",),
        field_names=("u",),
        B_red=np.array([[2.0, -3.0]], dtype=float),
        b_eff=np.array([0.0], dtype=float),
        project_feasible=(True,),
    )
    lam = np.array([0.5], dtype=float)
    row_scale = np.array([10.0, 4.0], dtype=float)

    stat = solver._vi_stationarity_residual(R_red, eq_data, lam, row_scale_red=row_scale)

    expected = R_red + row_scale * np.array([1.0, -1.5], dtype=float)
    assert np.allclose(stat, expected, atol=1.0e-14)


def test_internal_pdas_augmented_system_scales_upper_right_only() -> None:
    solver = _build_scalar_vi_solver(vi_params=VIParameters(equation_row_scaling=True))
    A_x = sp.csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float))
    rhs_x = np.array([5.0, 6.0], dtype=float)
    active_mask = np.array([False, True], dtype=bool)
    eq_data = _PreparedLinearEqualities(
        names=("mass",),
        field_names=("u",),
        B_red=np.array([[7.0, 8.0]], dtype=float),
        b_eff=np.array([0.0], dtype=float),
        project_feasible=(True,),
    )
    eq_res = np.array([9.0], dtype=float)
    row_scale = np.array([10.0, 20.0], dtype=float)

    A_aug, rhs_aug = solver._vi_build_augmented_system(
        A_x, rhs_x, active_mask, eq_data, eq_res, row_scale_red=row_scale
    )
    A_dense = np.asarray(A_aug.todense(), dtype=float)

    assert np.allclose(A_dense[:2, :2], np.asarray(A_x.todense(), dtype=float))
    assert np.allclose(A_dense[:2, 2:], np.array([[70.0], [0.0]], dtype=float))
    assert np.allclose(A_dense[2:, :2], np.array([[7.0, 8.0]], dtype=float))
    assert np.allclose(rhs_aug, np.array([5.0, 6.0, -9.0], dtype=float))


def test_internal_pdas_augmented_system_scales_x_block_and_equality_rows_under_column_scaling() -> None:
    solver = _build_scalar_vi_solver(vi_params=VIParameters(variable_column_scaling=True))
    A_x = sp.csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float))
    rhs_x = np.array([5.0, 6.0], dtype=float)
    active_mask = np.array([False, True], dtype=bool)
    eq_data = _PreparedLinearEqualities(
        names=("mass",),
        field_names=("u",),
        B_red=np.array([[7.0, 8.0]], dtype=float),
        b_eff=np.array([0.0], dtype=float),
        project_feasible=(True,),
    )
    eq_res = np.array([9.0], dtype=float)
    col_scale = np.array([10.0, 0.5], dtype=float)

    A_aug, rhs_aug = solver._vi_build_augmented_system(
        A_x, rhs_x, active_mask, eq_data, eq_res, col_scale_red=col_scale
    )
    A_dense = np.asarray(A_aug.todense(), dtype=float)

    assert np.allclose(A_dense[:2, :2], np.array([[10.0, 1.0], [30.0, 2.0]], dtype=float))
    assert np.allclose(A_dense[:2, 2:], np.array([[7.0], [0.0]], dtype=float))
    assert np.allclose(A_dense[2:, :2], np.array([[70.0, 4.0]], dtype=float))
    assert np.allclose(rhs_aug, np.array([5.0, 6.0, -9.0], dtype=float))


def test_internal_pdas_split_augmented_step_recovers_physical_step_after_column_scaling() -> None:
    solver = _build_scalar_vi_solver(vi_params=VIParameters(variable_column_scaling=True))
    eq_data = _PreparedLinearEqualities(
        names=("mass",),
        field_names=("u",),
        B_red=np.array([[1.0, 1.0]], dtype=float),
        b_eff=np.array([0.0], dtype=float),
        project_feasible=(True,),
    )

    dy_red, dlam = solver._vi_split_augmented_step(
        np.array([2.0, -1.0, 3.0], dtype=float),
        eq_data,
        col_scale_red=np.array([10.0, 0.5], dtype=float),
    )

    assert np.allclose(dy_red, np.array([20.0, -0.5], dtype=float))
    assert np.allclose(dlam, np.array([3.0], dtype=float))


def test_internal_pdas_selected_field_mask_marks_only_requested_fields() -> None:
    solver, *_ = _build_two_field_vi_solver(vi_params=VIParameters())
    mask = solver._vi_selected_field_mask(("alpha",))
    assert np.array_equal(mask, np.asarray(solver._vi_red_field_names, dtype=object) == "alpha")


def test_internal_pdas_fixed_state_semismooth_residual_uses_given_working_set() -> None:
    solver = _build_scalar_vi_solver(vi_params=VIParameters())
    G, act_lo, act_hi = solver._vi_semismooth_residual_from_state(
        x_red=np.array([0.2, 0.8, 0.4], dtype=float),
        stationarity_red=np.array([3.0, -2.0, 5.0], dtype=float),
        lo_red=np.array([0.0, 0.0, -np.inf], dtype=float),
        hi_red=np.array([1.0, 1.0, np.inf], dtype=float),
        state=np.array([1, -1, 0], dtype=np.int8),
    )

    np.testing.assert_array_equal(act_lo, np.array([True, False, False], dtype=bool))
    np.testing.assert_array_equal(act_hi, np.array([False, True, False], dtype=bool))
    np.testing.assert_allclose(G, np.array([0.2, -0.2, 5.0], dtype=float))


def test_internal_pdas_field_proximal_adds_diagonal_only_on_selected_fields() -> None:
    solver, dh, *_ = _build_two_field_vi_solver(vi_params=VIParameters())
    n = int(dh.total_dofs)
    diag = np.arange(1, n + 1, dtype=float)
    A = sp.diags(diag, format="csr")
    mask = solver._vi_selected_field_mask(("alpha",))
    A_prox, add = solver._vi_apply_field_proximal(A, A, mask, 2.0)

    expected_add = np.zeros((n,), dtype=float)
    expected_add[mask] = 2.0 * diag[mask]
    assert np.allclose(add, expected_add)
    assert np.allclose(np.asarray(A_prox.diagonal(), dtype=float), diag + expected_add)


def test_internal_pdas_ptc_adds_row_normalized_selected_block_by_default() -> None:
    solver, dh, *_ = _build_two_field_vi_solver(vi_params=VIParameters())
    n = int(dh.total_dofs)
    dense = np.eye(n, dtype=float)
    mask = solver._vi_selected_field_mask(("phi",))
    idx = np.flatnonzero(mask)
    assert idx.size >= 4
    block = np.eye(idx.size, dtype=float)
    block[0, :2] = np.array([2.0, -4.0], dtype=float)
    block[1, :2] = np.array([1.0, 3.0], dtype=float)
    dense[np.ix_(idx, idx)] = block
    A = sp.csr_matrix(dense)
    A_ptc, add = solver._vi_apply_ptc_regularization(A, A, mask, 3.0)

    added = np.asarray((A_ptc - A).todense(), dtype=float)
    expected_block = np.zeros((idx.size, idx.size), dtype=float)
    expected_block[0, :2] = 3.0 * np.array([0.5, -1.0], dtype=float)
    expected_block[1, :2] = 3.0 * np.array([1.0 / 3.0, 1.0], dtype=float)
    expected_block[2, 2] = 3.0
    expected_block[3, 3] = 3.0
    np.testing.assert_allclose(added[np.ix_(idx, idx)], expected_block)
    np.testing.assert_allclose(added[~mask, :], 0.0)
    np.testing.assert_allclose(added[:, ~mask], 0.0)
    expected_add = np.zeros((n,), dtype=float)
    expected_add[idx] = np.array([1.5, 3.0, 3.0, 3.0], dtype=float)
    np.testing.assert_allclose(add, expected_add)


def test_internal_pdas_ptc_diag_mode_retains_legacy_diagonal_regularization() -> None:
    solver, dh, *_ = _build_two_field_vi_solver(
        vi_params=VIParameters(ptc_operator_mode="diag")
    )
    n = int(dh.total_dofs)
    diag = np.arange(1, n + 1, dtype=float)
    A = sp.diags(diag, format="csr")
    mask = solver._vi_selected_field_mask(("phi",))
    A_ptc, add = solver._vi_apply_ptc_regularization(A, A, mask, 3.0)

    expected_add = np.zeros((n,), dtype=float)
    expected_add[mask] = 3.0 * diag[mask]
    np.testing.assert_allclose(add, expected_add)
    np.testing.assert_allclose(np.asarray(A_ptc.diagonal(), dtype=float), diag + expected_add)


def test_internal_pdas_selected_field_freeze_zeroes_complement_rhs() -> None:
    solver, dh, *_ = _build_two_field_vi_solver(vi_params=VIParameters())
    n = int(dh.total_dofs)
    A = sp.csr_matrix(np.arange(1, n * n + 1, dtype=float).reshape((n, n)))
    rhs = np.arange(1, n + 1, dtype=float)
    mask = solver._vi_selected_field_mask(("alpha",))

    A_frozen, rhs_frozen = solver._vi_apply_selected_field_freeze(A, rhs, mask)

    frozen = np.flatnonzero(~mask)
    assert np.allclose(rhs_frozen[frozen], 0.0)
    assert np.allclose(np.asarray(A_frozen[frozen, :].todense()), np.eye(n, dtype=float)[frozen, :])


def test_internal_pdas_field_proximal_recovery_trigger_requires_stable_active_set_and_small_gap() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            field_proximal_recovery=True,
            field_proximal_recovery_stable_iters=1,
            field_proximal_recovery_ginf_trigger=5.0e-2,
            field_proximal_recovery_gap_ratio=1.0,
            field_proximal_recovery_eq_abs=1.0e-10,
        )
    )

    assert solver._vi_should_try_field_proximal_recovery(
        metrics={
            "G_inf": 1.0e-1,
            "inactive_res_inf": 2.0e-1,
            "active_gap_inf": 1.0e-12,
            "equality_inf": 1.0e-12,
        },
        active_stable_count=1,
    )
    assert not solver._vi_should_try_field_proximal_recovery(
        metrics={
            "G_inf": 1.0e-1,
            "inactive_res_inf": 2.0e-1,
            "active_gap_inf": 3.0e-1,
            "equality_inf": 1.0e-12,
        },
        active_stable_count=1,
    )
    assert not solver._vi_should_try_field_proximal_recovery(
        metrics={
            "G_inf": 1.0e-2,
            "inactive_res_inf": 2.0e-1,
            "active_gap_inf": 1.0e-12,
            "equality_inf": 1.0e-12,
        },
        active_stable_count=1,
    )


def test_internal_pdas_ptc_recovery_trigger_requires_stable_active_set_and_small_gap() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            ptc_recovery=True,
            ptc_stable_iters=1,
            ptc_ginf_trigger=5.0e-2,
            ptc_gap_ratio=1.0,
            ptc_eq_abs=1.0e-10,
        )
    )

    assert solver._vi_should_try_ptc_recovery(
        metrics={
            "G_inf": 1.0e-1,
            "inactive_res_inf": 2.0e-1,
            "active_gap_inf": 1.0e-12,
            "equality_inf": 1.0e-12,
        },
        active_stable_count=1,
    )
    assert not solver._vi_should_try_ptc_recovery(
        metrics={
            "G_inf": 1.0e-1,
            "inactive_res_inf": 2.0e-1,
            "active_gap_inf": 3.0e-1,
            "equality_inf": 1.0e-12,
        },
        active_stable_count=1,
    )


def test_internal_pdas_field_proximal_arms_after_slow_contraction_on_identified_set() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            field_proximal_recovery=True,
            field_proximal_recovery_alpha_trigger=0.25,
            field_proximal_recovery_merit_ratio_trigger=0.8,
            field_proximal_recovery_g_ratio_trigger=0.8,
        )
    )

    assert solver._vi_should_arm_field_proximal_after_accept(
        accepted_alpha=0.2,
        merit_ratio=0.6,
        accepted_g_ratio=0.5,
    )
    assert solver._vi_should_arm_field_proximal_after_accept(
        accepted_alpha=0.6,
        merit_ratio=0.85,
        accepted_g_ratio=0.5,
    )
    assert solver._vi_should_arm_field_proximal_after_accept(
        accepted_alpha=0.6,
        merit_ratio=0.5,
        accepted_g_ratio=0.9,
    )
    assert not solver._vi_should_arm_field_proximal_after_accept(
        accepted_alpha=0.6,
        merit_ratio=0.5,
        accepted_g_ratio=0.5,
    )


def test_internal_pdas_field_proximal_identified_window_trigger() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            field_proximal_recovery=True,
            field_proximal_recovery_identified_window=True,
            field_proximal_recovery_stable_iters=1,
            field_proximal_recovery_ginf_trigger=5.0e-2,
            field_proximal_recovery_ginf_max=2.0e-1,
            field_proximal_recovery_gap_ratio=1.0,
            field_proximal_recovery_eq_abs=1.0e-10,
        )
    )

    assert solver._vi_should_force_field_proximal_on_identified_window(
        metrics={
            "G_inf": 1.0e-1,
            "inactive_res_inf": 1.5e-1,
            "active_gap_inf": 1.0e-12,
            "equality_inf": 1.0e-12,
        },
        active_stable_count=1,
        changed=0,
    )
    assert not solver._vi_should_force_field_proximal_on_identified_window(
        metrics={
            "G_inf": 3.0e-1,
            "inactive_res_inf": 3.0e-1,
            "active_gap_inf": 1.0e-12,
            "equality_inf": 1.0e-12,
        },
        active_stable_count=1,
        changed=0,
    )
    assert solver._vi_should_force_field_proximal_on_identified_window(
        metrics={
            "G_inf": 1.0e-1,
            "inactive_res_inf": 1.5e-1,
            "active_gap_inf": 1.0e-12,
            "equality_inf": 1.0e-12,
        },
        active_stable_count=0,
        changed=0,
    )


def test_internal_pdas_prefers_field_proximal_phase_once_lambda_is_armed() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            field_proximal_recovery=True,
        )
    )
    prox_mask = np.array([True, False, True, False], dtype=bool)
    solver._vi_field_prox_lambda_current = 0.0
    assert not solver._vi_should_prefer_field_proximal_phase(prox_mask=prox_mask)
    solver._vi_field_prox_lambda_current = 1.0e-2
    assert solver._vi_should_prefer_field_proximal_phase(prox_mask=prox_mask)


def test_internal_pdas_prefers_ptc_phase_once_sigma_is_armed() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            ptc_recovery=True,
        )
    )
    ptc_mask = np.array([True, False, True, False], dtype=bool)
    solver._vi_ptc_sigma_current = 0.0
    assert not solver._vi_should_prefer_ptc_phase(ptc_mask=ptc_mask)
    solver._vi_ptc_sigma_current = 1.0e-2
    assert solver._vi_should_prefer_ptc_phase(ptc_mask=ptc_mask)


def test_internal_pdas_freezes_local_trial_model_when_guard_or_prox_is_active() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            field_proximal_recovery=True,
            ptc_recovery=True,
        )
    )

    assert not solver._vi_should_freeze_local_trial_model()
    solver._vi_working_set_guard_active = np.ones((solver.active_dofs.size,), dtype=np.int8)
    assert solver._vi_should_freeze_local_trial_model()
    solver._vi_working_set_guard_active = None
    solver._vi_field_prox_lambda_current = 1.0e-2
    assert solver._vi_should_freeze_local_trial_model()
    solver._vi_field_prox_lambda_current = 0.0
    solver._vi_ptc_sigma_current = 1.0e-2
    assert solver._vi_should_freeze_local_trial_model()


def test_internal_pdas_arms_guard_after_slow_identified_accept() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            working_set_guard_after_affine=2,
            field_proximal_recovery=True,
            field_proximal_recovery_stable_iters=1,
            field_proximal_recovery_ginf_trigger=5.0e-2,
            field_proximal_recovery_gap_ratio=1.0,
            field_proximal_recovery_eq_abs=1.0e-10,
        )
    )

    assert solver._vi_should_arm_working_set_guard_after_accept(
        metrics={
            "G_inf": 1.0e-1,
            "inactive_res_inf": 1.2e-1,
            "active_gap_inf": 1.0e-12,
            "equality_inf": 1.0e-12,
        },
        active_stable_count=1,
        changed=0,
        accepted_alpha=1.0e-1,
        merit_ratio=0.95,
        accepted_g_ratio=0.92,
    )
    assert not solver._vi_should_arm_working_set_guard_after_accept(
        metrics={
            "G_inf": 1.0e-1,
            "inactive_res_inf": 1.2e-1,
            "active_gap_inf": 1.0e-12,
            "equality_inf": 1.0e-12,
        },
        active_stable_count=1,
        changed=2,
        accepted_alpha=1.0e-1,
        merit_ratio=0.95,
        accepted_g_ratio=0.92,
    )


def test_internal_pdas_updates_field_proximal_after_identified_accept() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            field_proximal_recovery=True,
            field_proximal_recovery_stable_iters=1,
            field_proximal_recovery_ginf_trigger=5.0e-2,
            field_proximal_recovery_gap_ratio=1.0,
            field_proximal_recovery_eq_abs=1.0e-10,
            field_proximal_recovery_lambda0=1.0e-2,
            field_proximal_recovery_lambda_max=1.0e2,
            field_proximal_recovery_growth=5.0,
            field_proximal_recovery_identified_window=True,
            field_proximal_recovery_ginf_max=2.0e-1,
        )
    )

    prox_mask = np.array([True, False, True, False], dtype=bool)
    solver._vi_update_field_proximal_after_accept(
        prox_mask=prox_mask,
        metrics={
            "G_inf": 1.0e-1,
            "inactive_res_inf": 1.2e-1,
            "active_gap_inf": 1.0e-12,
            "equality_inf": 1.0e-12,
        },
        active_stable_count=1,
        changed=0,
        accepted_alpha=1.0,
        merit_ratio=0.95,
        accepted_g_ratio=0.91,
    )

    assert solver._vi_field_prox_lambda_current == pytest.approx(1.0e-2)


def test_internal_pdas_updates_ptc_after_identified_accept() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            ptc_recovery=True,
            ptc_stable_iters=1,
            ptc_ginf_trigger=5.0e-2,
            ptc_gap_ratio=1.0,
            ptc_eq_abs=1.0e-10,
            ptc_sigma0=5.0e-2,
            ptc_sigma_max=1.0e2,
            ptc_growth=5.0,
            ptc_identified_window=True,
            ptc_ginf_max=2.0e-1,
        )
    )

    ptc_mask = np.array([True, False, True, False], dtype=bool)
    solver._vi_update_ptc_after_accept(
        ptc_mask=ptc_mask,
        metrics={
            "G_inf": 1.0e-1,
            "inactive_res_inf": 1.2e-1,
            "active_gap_inf": 1.0e-12,
            "equality_inf": 1.0e-12,
        },
        active_stable_count=1,
        changed=0,
        accepted_alpha=1.0,
        merit_ratio=0.95,
        accepted_g_ratio=0.91,
    )

    assert solver._vi_ptc_sigma_current == pytest.approx(5.0e-2)


def test_internal_pdas_keeps_existing_ptc_sigma_after_weak_accept() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            ptc_recovery=True,
            ptc_stable_iters=1,
            ptc_ginf_trigger=5.0e-2,
            ptc_gap_ratio=1.0,
            ptc_eq_abs=1.0e-10,
            ptc_sigma0=5.0e-2,
            ptc_sigma_max=1.0e2,
            ptc_growth=5.0,
            ptc_decay=0.5,
            ptc_identified_window=True,
            ptc_ginf_max=2.0e-1,
        )
    )

    solver._vi_ptc_sigma_current = 2.5e-1
    ptc_mask = np.array([True, False, True, False], dtype=bool)
    solver._vi_update_ptc_after_accept(
        ptc_mask=ptc_mask,
        metrics={
            "G_inf": 1.0e-1,
            "inactive_res_inf": 1.2e-1,
            "active_gap_inf": 1.0e-12,
            "equality_inf": 1.0e-12,
        },
        active_stable_count=1,
        changed=0,
        accepted_alpha=1.0,
        merit_ratio=0.98,
        accepted_g_ratio=0.99,
    )

    assert solver._vi_ptc_sigma_current == pytest.approx(2.5e-1)


def test_internal_pdas_builds_anderson_candidate_from_history() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            anderson_acceleration=True,
            anderson_history=2,
            anderson_regularization=1.0e-10,
            anderson_damping=1.0,
        )
    )

    solver._vi_record_anderson_history(
        x_prev=np.array([0.0], dtype=float),
        lambda_prev=np.zeros((0,), dtype=float),
        x_next=np.array([1.0], dtype=float),
        lambda_next=np.zeros((0,), dtype=float),
    )
    cand = solver._vi_build_anderson_candidate(
        x_curr=np.array([1.0], dtype=float),
        lambda_curr=np.zeros((0,), dtype=float),
        x_proposed=np.array([0.25], dtype=float),
        lambda_proposed=np.zeros((0,), dtype=float),
    )

    assert cand is not None
    step, lam_target = cand
    assert step.shape == (1,)
    assert lam_target.shape == (0,)
    assert np.isfinite(step[0])


def test_internal_pdas_import_vi_state_maps_transport_state_into_monolithic_space() -> None:
    transport_solver, *_ = _build_two_field_vi_solver(
        vi_params=VIParameters(c=1.0, enter_tol=1.0e-6, leave_tol=1.0e-6)
    )
    transport_solver.set_active_fields(["alpha", "phi"])
    prev_state = np.zeros((transport_solver.active_dofs.size,), dtype=np.int8)
    alpha_mask = np.asarray(transport_solver._vi_red_field_names, dtype=object) == "alpha"
    prev_state[alpha_mask] = 1
    transport_solver._vi_prev_state = prev_state.copy()
    transport_solver._vi_pending_state = prev_state.copy()
    transport_solver._vi_pending_count = np.zeros(prev_state.shape, dtype=np.int16)
    transport_solver._vi_forced_state = prev_state.copy()
    transport_solver._vi_working_set_guard_state = prev_state.copy()
    transport_solver._vi_working_set_guard_remaining = 2
    transport_solver._vi_working_set_guard_active = None
    exported = transport_solver.export_vi_state()

    monolithic_solver, *_ = _build_two_field_vi_solver(
        vi_params=VIParameters(c=1.0, enter_tol=1.0e-6, leave_tol=1.0e-6)
    )
    assert monolithic_solver.import_vi_state(exported, force_once=True)

    imported_prev = np.asarray(monolithic_solver._vi_prev_state, dtype=np.int8)
    imported_forced = np.asarray(monolithic_solver._vi_forced_state, dtype=np.int8)
    imported_guard = np.asarray(monolithic_solver._vi_working_set_guard_state, dtype=np.int8)
    mono_alpha_mask = np.asarray(monolithic_solver._vi_red_field_names, dtype=object) == "alpha"
    mono_phi_mask = np.asarray(monolithic_solver._vi_red_field_names, dtype=object) == "phi"

    assert np.all(imported_prev[mono_alpha_mask] == 1)
    assert np.all(imported_prev[mono_phi_mask] == 0)
    assert np.all(imported_forced[mono_alpha_mask] == 1)
    assert np.all(imported_forced[mono_phi_mask] == 0)
    assert np.all(imported_guard[mono_alpha_mask] == 1)
    assert np.all(imported_guard[mono_phi_mask] == 0)
    assert monolithic_solver._vi_working_set_guard_remaining == 2
    assert monolithic_solver._vi_preserve_state_on_next_solve is True


def test_internal_pdas_detects_period2_state_cycle() -> None:
    solver = _build_scalar_vi_solver(vi_params=VIParameters())
    info = solver._vi_detect_period2_cycle(
        current_state=np.array([1, 0, -1, 0], dtype=np.int8),
        prev_state=np.array([0, 0, -1, 1], dtype=np.int8),
        prevprev_state=np.array([1, 0, -1, 0], dtype=np.int8),
    )
    assert info is not None
    assert int(info["period"]) == 2
    assert int(info["nflip"]) == 2
    assert np.array_equal(np.asarray(info["flip_idx"], dtype=int), np.array([0, 3], dtype=int))


def test_internal_pdas_affine_cycle_fallback_reduces_vi_residual_on_box_problem() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            c=10.0,
            affine_cycle_fallback=True,
            affine_cycle_fallback_max_it=6,
        )
    )
    nred = int(solver.active_dofs.size)
    A_red = sp.eye(nred, format="csr") * 2.0
    x0_red = np.full((nred,), 1.2, dtype=float)
    lo_red, hi_red = solver._bounds_reduced()
    c_red = np.full((nred,), 10.0, dtype=float)
    F0 = np.full((nred,), 0.5, dtype=float)
    rhs_aff_red = np.asarray(A_red @ x0_red, dtype=float).ravel() - F0
    eq_data = _PreparedLinearEqualities(
        names=tuple(),
        field_names=tuple(),
        B_red=np.zeros((0, nred), dtype=float),
        b_eff=np.zeros((0,), dtype=float),
        project_feasible=tuple(),
    )

    F_init, G_init, act_lo_init, act_hi_init, eq_init = solver._vi_affine_residual_and_sets(
        A_red=A_red,
        rhs_aff_red=rhs_aff_red,
        x_red=x0_red,
        lambda_eq=np.zeros((0,), dtype=float),
        lo_red=lo_red,
        hi_red=hi_red,
        c_red=c_red,
        eq_data=eq_data,
    )
    metrics_init = solver._vi_metrics(
        x0_red,
        F_init,
        lo_red,
        hi_red,
        act_lo_init,
        act_hi_init,
        G_init,
        eq_init,
    )

    result = solver._vi_affine_cycle_fallback(
        A_red=A_red,
        rhs_aff_red=rhs_aff_red,
        x0_red=x0_red,
        lambda0_eq=np.zeros((0,), dtype=float),
        lo_red=lo_red,
        hi_red=hi_red,
        c_red=c_red,
        eq_data=eq_data,
        tol=1.0e-10,
    )

    assert result is not None
    y_new, lam_new, metrics_new = result
    assert lam_new.size == 0
    assert float(metrics_new["G_inf"]) < float(metrics_init["G_inf"])
    assert np.all(np.asarray(y_new, dtype=float) <= 1.0 + 1.0e-12)


def test_internal_pdas_affine_cycle_fallback_accepts_small_merit_improvement_on_ill_scaled_step() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            c=10.0,
            affine_cycle_fallback=True,
            affine_cycle_fallback_max_it=6,
        )
    )
    nred = int(solver.active_dofs.size)
    A_red = sp.eye(nred, format="csr") * 1.2742749857031334e-04
    x0_red = np.full((nred,), 0.8, dtype=float)
    lo_red, hi_red = solver._bounds_reduced()
    c_red = np.full((nred,), 10.0, dtype=float)
    F0 = np.full((nred,), 2.0e-2, dtype=float)
    rhs_aff_red = np.asarray(A_red @ x0_red, dtype=float).ravel() - F0
    eq_data = _PreparedLinearEqualities(
        names=tuple(),
        field_names=tuple(),
        B_red=np.zeros((0, nred), dtype=float),
        b_eff=np.zeros((0,), dtype=float),
        project_feasible=tuple(),
    )

    F_init, G_init, act_lo_init, act_hi_init, eq_init = solver._vi_affine_residual_and_sets(
        A_red=A_red,
        rhs_aff_red=rhs_aff_red,
        x_red=x0_red,
        lambda_eq=np.zeros((0,), dtype=float),
        lo_red=lo_red,
        hi_red=hi_red,
        c_red=c_red,
        eq_data=eq_data,
    )
    metrics_init = solver._vi_metrics(
        x0_red,
        F_init,
        lo_red,
        hi_red,
        act_lo_init,
        act_hi_init,
        G_init,
        eq_init,
    )

    result = solver._vi_affine_cycle_fallback(
        A_red=A_red,
        rhs_aff_red=rhs_aff_red,
        x0_red=x0_red,
        lambda0_eq=np.zeros((0,), dtype=float),
        lo_red=lo_red,
        hi_red=hi_red,
        c_red=c_red,
        eq_data=eq_data,
        tol=1.0e-12,
    )

    assert result is not None
    y_new, lam_new, metrics_new = result
    assert lam_new.size == 0
    assert float(metrics_new["G_half"]) < float(metrics_init["G_half"])
    assert float(metrics_new["G_inf"]) < float(metrics_init["G_inf"])
    assert np.all(np.isfinite(np.asarray(y_new, dtype=float)))


def test_internal_pdas_affine_identified_acceleration_gate() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            affine_identified_acceleration=True,
            affine_identified_stable_iters=2,
            affine_identified_ginf_trigger=5.0e-2,
            affine_identified_gap_ratio=1.0,
            affine_identified_eq_abs=1.0e-10,
        )
    )
    metrics = {
        "G_inf": 1.0e-2,
        "inactive_res_inf": 1.0e-2,
        "active_gap_inf": 1.0e-12,
        "equality_inf": 1.0e-13,
    }
    assert solver._vi_should_try_affine_identified_acceleration(
        metrics=metrics,
        changed=0,
        active_stable_count=2,
    )
    assert not solver._vi_should_try_affine_identified_acceleration(
        metrics=metrics,
        changed=1,
        active_stable_count=2,
    )
    metrics_bad = dict(metrics)
    metrics_bad["G_inf"] = 1.0e-1
    assert not solver._vi_should_try_affine_identified_acceleration(
        metrics=metrics_bad,
        changed=0,
        active_stable_count=2,
    )


def test_direct_solve_quality_check_is_looser_for_vi_augmented_context(monkeypatch) -> None:
    solver = _build_scalar_vi_solver(vi_params=VIParameters())
    A = sp.eye(int(solver.active_dofs.size), format="csr")
    rhs = np.ones((A.shape[0],), dtype=float)
    sol = np.ones((A.shape[0],), dtype=float)

    monkeypatch.setattr(
        solver,
        "_linear_solution_quality",
        lambda A_in, rhs_in, sol_in: (1.0e-6, 1.0e-6),
    )

    assert not solver._direct_solve_quality_ok(A, rhs, sol, solver_name="scipy")
    solver._linear_solve_context = "vi_augmented"
    try:
        assert solver._direct_solve_quality_ok(A, rhs, sol, solver_name="scipy")
    finally:
        solver._linear_solve_context = None


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


def test_internal_pdas_default_active_step_respects_backtracking_alpha() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            active_step_delta_active_trigger=0,
            active_step_soft_alpha=1.0,
        )
    )
    S_red = np.array([1.0, -2.0, 3.0, -4.0], dtype=float)
    active = np.array([True, True, False, True], dtype=bool)

    step, used_soft = solver._vi_apply_active_step_policy(
        alpha=0.125,
        S_red=S_red,
        active_mask=active,
        strong_active_mask=None,
        active_delta_ref=0,
    )

    assert not used_soft
    assert np.allclose(step, 0.125 * S_red)


def test_internal_pdas_predicted_bound_activity_detects_new_actives() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            enter_tol=1.0e-6,
            leave_tol=1.0e-6,
        )
    )
    x_red = np.array([0.5, 0.5, 0.5], dtype=float)
    step_red = np.array([0.6, -0.55, 0.1], dtype=float)
    lo_red = np.zeros(3, dtype=float)
    hi_red = np.ones(3, dtype=float)
    act_lo = np.zeros(3, dtype=bool)
    act_hi = np.zeros(3, dtype=bool)

    predicted, entering, predicted_delta = solver._vi_predicted_bound_activity(
        x_red=x_red,
        step_red=step_red,
        lo_red=lo_red,
        hi_red=hi_red,
        act_lo=act_lo,
        act_hi=act_hi,
    )

    assert predicted_delta == 2
    assert np.array_equal(predicted, np.array([True, True, False], dtype=bool))
    assert np.array_equal(entering, np.array([True, True, False], dtype=bool))


def test_internal_pdas_predict_sets_from_state_matches_indicator_logic() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            c=10.0,
            enter_tol=1.0e-6,
            leave_tol=1.0e-6,
        )
    )
    x_red = np.array([0.2, 0.8, 0.5], dtype=float)
    lo_red = np.zeros(3, dtype=float)
    hi_red = np.ones(3, dtype=float)
    c_red = np.full(3, 10.0, dtype=float)
    stationarity = np.array([3.0, -3.0, 0.0], dtype=float)

    act_lo, act_hi = solver._vi_predict_sets_from_state(
        x_red=x_red,
        stationarity_red=stationarity,
        lo_red=lo_red,
        hi_red=hi_red,
        c_red=c_red,
        prev_state=np.zeros(3, dtype=np.int8),
    )

    assert np.array_equal(act_lo, np.array([True, False, False], dtype=bool))
    assert np.array_equal(act_hi, np.array([False, True, False], dtype=bool))


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


def test_internal_pdas_unconstrained_lm_uses_row_scale_for_zero_diagonal_rows() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            unconstrained_lm=True,
            unconstrained_lm_lambda0=0.5,
            unconstrained_lm_min_diag=1.0e-12,
        )
    )
    A = sp.csr_matrix(np.array([[0.0, 2.0], [3.0, 4.0]], dtype=float))
    R = np.array([1.0, -1.0], dtype=float)

    H, rhs, diag = solver._vi_build_unconstrained_lm_system(A, R, 0.5)

    assert np.allclose(diag, np.array([2.0, 4.0]))
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


def test_internal_pdas_reports_small_semismooth_residual_with_nonzero_raw_residual_on_active_bounds() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(c=2.0, enter_tol=0.0, leave_tol=0.0)
    )
    lo_red, hi_red = solver._bounds_reduced()
    x_red = lo_red.copy()
    R_red = np.full(x_red.shape, 0.25, dtype=float)

    G_red, act_lo, act_hi = solver._pdas_residual_and_sets(x_red, R_red, lo_red, hi_red)
    metrics = solver._vi_metrics(x_red, R_red, lo_red, hi_red, act_lo, act_hi, G_red)

    assert np.all(act_lo)
    assert not np.any(act_hi)
    assert np.isclose(float(np.linalg.norm(R_red, ord=np.inf)), 0.25)
    assert np.isclose(float(np.linalg.norm(G_red, ord=np.inf)), 0.0)
    assert np.isclose(metrics["G_inf"], 0.0)
    assert np.isclose(metrics["inactive_res_inf"], 0.0)
    assert np.isclose(metrics["active_gap_inf"], 0.0)


def test_internal_pdas_multifield_feasible_projection_supports_dynamic_phi_weights() -> None:
    solver, dh, alpha_k, alpha_n, phi_k, phi_n = _build_two_field_vi_solver(
        vi_params=VIParameters(c=1.0, project_initial_guess=True, project_each_iteration=False)
    )

    alpha_k.nodal_values[:] = np.array([1.2, -0.1, 0.8, 1.1], dtype=float)
    phi_k.nodal_values[:] = np.array([1.2, -0.2, 0.4, 0.9], dtype=float)
    alpha_n.nodal_values[:] = np.array([0.6, 0.4, 0.7, 0.5], dtype=float)
    phi_n.nodal_values[:] = np.array([0.3, 0.2, 0.4, 0.1], dtype=float)

    alpha_slice = np.asarray(dh.get_field_slice("alpha"), dtype=int).ravel()
    phi_slice = np.asarray(dh.get_field_slice("phi"), dtype=int).ravel()

    alpha_weights = np.zeros((int(dh.total_dofs),), dtype=float)
    alpha_weights[alpha_slice] = 1.0

    def _phi_weights(*, funcs, **_kwargs):
        alpha_cur = alpha_k
        for f in list(funcs or []):
            if str(getattr(f, "name", "")) == "alpha_k":
                alpha_cur = f
                break
        w = np.zeros((int(dh.total_dofs),), dtype=float)
        w[phi_slice] = np.asarray(alpha_cur.nodal_values, dtype=float)
        return w

    def _alpha_target(*, prev_funcs, **_kwargs):
        alpha_prev = alpha_n
        for f in list(prev_funcs or []):
            if str(getattr(f, "name", "")) == "alpha_n":
                alpha_prev = f
                break
        return float(np.sum(np.asarray(alpha_prev.nodal_values, dtype=float)))

    def _phi_target(*, prev_funcs, **_kwargs):
        alpha_prev = alpha_n
        phi_prev = phi_n
        for f in list(prev_funcs or []):
            if str(getattr(f, "name", "")) == "alpha_n":
                alpha_prev = f
            elif str(getattr(f, "name", "")) == "phi_n":
                phi_prev = f
        return float(
            np.asarray(alpha_prev.nodal_values, dtype=float) @ np.asarray(phi_prev.nodal_values, dtype=float)
        )

    solver.set_linear_equalities(
        [
            LinearEqualityConstraint(
                name="alpha_mass",
                weights_full=alpha_weights,
                target_callback=_alpha_target,
                field_name="alpha",
                project_feasible=True,
            ),
            LinearEqualityConstraint(
                name="phi_weighted_mass",
                weights_callback=_phi_weights,
                target_callback=_phi_target,
                field_name="phi",
                project_feasible=True,
            ),
        ]
    )

    funcs = [alpha_k, phi_k]
    prev_funcs = [alpha_n, phi_n]
    lo_red, hi_red = solver._bounds_reduced()
    eq_prepare = lambda: solver._vi_prepare_linear_equalities(funcs, prev_funcs, {}, [])
    eq_data = eq_prepare()
    solver._project_funcs_to_bounds(funcs, [], lo_red, hi_red, eq_data=eq_data, eq_prepare_callback=eq_prepare)
    eq_final = eq_prepare()
    x_final = solver._pack_reduced_iterate(funcs)
    eq_res = solver._vi_equality_residual(x_final, eq_final)

    assert np.all(np.asarray(alpha_k.nodal_values, dtype=float) >= -1.0e-12)
    assert np.all(np.asarray(alpha_k.nodal_values, dtype=float) <= 1.0 + 1.0e-12)
    assert np.all(np.asarray(phi_k.nodal_values, dtype=float) >= -1.0e-12)
    assert np.all(np.asarray(phi_k.nodal_values, dtype=float) <= 1.0 + 1.0e-12)
    assert float(np.linalg.norm(eq_res, ord=np.inf)) <= 1.0e-10
    assert np.isclose(
        float(np.sum(np.asarray(alpha_k.nodal_values, dtype=float))),
        float(np.sum(np.asarray(alpha_n.nodal_values, dtype=float))),
        atol=1.0e-10,
    )
    assert np.isclose(
        float(np.asarray(alpha_k.nodal_values, dtype=float) @ np.asarray(phi_k.nodal_values, dtype=float)),
        float(np.asarray(alpha_n.nodal_values, dtype=float) @ np.asarray(phi_n.nodal_values, dtype=float)),
        atol=1.0e-10,
    )


def test_internal_pdas_vi_line_search_rejects_unfiltered_best_merit_step(monkeypatch) -> None:
    solver, _, _, _, _, _ = _build_two_field_vi_solver(
        vi_params=VIParameters(
            c=1.0,
            merit_mode="split",
            filter_max_ginf_growth=1.0,
            project_each_iteration=False,
        )
    )
    lo_red, hi_red = solver._bounds_reduced()
    nred = int(lo_red.size)
    state = {"x": np.zeros((nred,), dtype=float)}
    direction = np.zeros((nred,), dtype=float)
    direction[:2] = 1.0

    def _residual_for(x_vec: np.ndarray) -> np.ndarray:
        alpha = float(np.max(np.asarray(x_vec[:2], dtype=float))) if x_vec.size else 0.0
        out = np.zeros((nred,), dtype=float)
        if alpha > 0.0:
            out[0] = 1.0 + 0.2 * alpha
            out[1] = 0.2 * (1.0 - alpha)
        else:
            out[0] = 1.0
            out[1] = 1.0
        return out

    monkeypatch.setattr(solver.restrictor, "expand_vec", lambda v: np.asarray(v, dtype=float).copy())
    monkeypatch.setattr(solver.dh, "add_to_functions", lambda step, funcs: state.__setitem__("x", state["x"] + np.asarray(step, dtype=float)))
    monkeypatch.setattr(solver.dh, "apply_bcs", lambda bcs, *funcs: None)
    monkeypatch.setattr(solver, "_assemble_system_reduced", lambda coeffs, need_matrix=False: (None, _residual_for(state["x"])))
    monkeypatch.setattr(solver, "_pack_reduced_iterate", lambda funcs: state["x"].copy())
    monkeypatch.setattr(solver, "_vi_stationarity_residual", lambda R_red, eq_data, lam_eq: np.asarray(R_red, dtype=float))
    monkeypatch.setattr(
        solver,
        "_pdas_residual_and_sets",
        lambda x_red, R_red, lo_red_in, hi_red_in, c_red=None: (
            np.asarray(R_red, dtype=float).copy(),
            np.zeros(x_red.shape, dtype=bool),
            np.zeros(x_red.shape, dtype=bool),
        ),
    )
    monkeypatch.setattr(solver, "_vi_equality_residual", lambda x_red, eq_data: np.zeros((0,), dtype=float))
    solver.np.ls_max_iter = 4
    solver.np.ls_reduction = 0.5
    solver.np.ls_mode = "dealii"

    x0_red = np.zeros((nred,), dtype=float)
    R0_red = _residual_for(x0_red)
    G0 = R0_red.copy()
    eq0_res = np.zeros((0,), dtype=float)
    act_lo0 = np.zeros((nred,), dtype=bool)
    act_hi0 = np.zeros((nred,), dtype=bool)
    lambda0_eq = np.zeros((0,), dtype=float)
    c_red = np.ones((nred,), dtype=float)

    class _EqData:
        def __init__(self, ncols: int):
            self.B_red = np.zeros((0, ncols), dtype=float)

    eq_data = _EqData(nred)

    with pytest.raises(RuntimeError, match="Line search failed"):
        solver._line_search_vi_reduced(
            x0_red=x0_red,
            R0_red=R0_red,
            G0=G0,
            eq0_res=eq0_res,
            act_lo0=act_lo0,
            act_hi0=act_hi0,
            lambda0_eq=lambda0_eq,
            S_red=direction,
            S_lam_eq=np.zeros((0,), dtype=float),
            c_red=c_red,
            eq_data=eq_data,
            funcs=[],
            coeffs={},
            bcs_now=[],
            lo_red=lo_red,
            hi_red=hi_red,
        )


def test_internal_pdas_projection_refreshes_dynamic_bounds_from_pre_cb() -> None:
    solver, _, alpha_k, _, phi_k, _ = _build_two_field_vi_solver(
        vi_params=VIParameters(
            c=1.0,
            project_each_iteration=False,
        )
    )
    solver.set_box_bounds(by_field={"alpha": (0.0, 1.0), "phi": (0.0, 2.0)})
    lo_red, hi_red = solver._bounds_reduced()

    alpha_k.nodal_values[:] = 0.0
    phi_k.nodal_values[:] = 1.5
    funcs = [alpha_k, phi_k]

    def _tighten_bounds(_funcs) -> None:
        solver.set_box_bounds(by_field={"alpha": (0.0, 1.0), "phi": (0.0, 1.0)})

    solver.pre_cb = _tighten_bounds

    nred = int(lo_red.size)

    def _empty_equalities() -> _PreparedLinearEqualities:
        return _PreparedLinearEqualities(
            names=tuple(),
            field_names=tuple(),
            B_red=np.zeros((0, nred), dtype=float),
            b_eff=np.zeros((0,), dtype=float),
            project_feasible=tuple(),
        )

    solver._project_funcs_to_bounds(
        funcs,
        [],
        lo_red,
        hi_red,
        eq_prepare_callback=_empty_equalities,
    )

    assert np.all(np.asarray(phi_k.nodal_values, dtype=float) <= 1.0 + 1.0e-12)


def test_internal_pdas_vi_line_search_forwards_eq_prepare_callback_to_projection(monkeypatch) -> None:
    solver, _, alpha_k, _, phi_k, _ = _build_two_field_vi_solver(
        vi_params=VIParameters(
            c=1.0,
            project_each_iteration=True,
        )
    )

    funcs = [alpha_k, phi_k]
    lo_red, hi_red = solver._bounds_reduced()
    nred = int(lo_red.size)
    sentinel = RuntimeError("eq_prepare_callback_forwarded")

    monkeypatch.setattr(solver.restrictor, "expand_vec", lambda v: np.asarray(v, dtype=float).copy())
    monkeypatch.setattr(solver.dh, "add_to_functions", lambda step, funcs_in: None)
    monkeypatch.setattr(solver.dh, "apply_bcs", lambda bcs, *funcs_in: None)

    def _project_with_assertion(funcs_in, bcs_in, lo_in, hi_in, *, eq_data=None, eq_prepare_callback=None):
        assert eq_prepare_callback is eq_prepare
        raise sentinel

    monkeypatch.setattr(solver, "_project_funcs_to_bounds", _project_with_assertion)

    eq_prepare = lambda: _PreparedLinearEqualities(
        names=tuple(),
        field_names=tuple(),
        B_red=np.zeros((0, nred), dtype=float),
        b_eff=np.zeros((0,), dtype=float),
        project_feasible=tuple(),
    )
    eq_data = eq_prepare()

    with pytest.raises(RuntimeError, match="eq_prepare_callback_forwarded"):
        solver._line_search_vi_reduced(
            x0_red=np.zeros((nred,), dtype=float),
            R0_red=np.zeros((nred,), dtype=float),
            G0=np.zeros((nred,), dtype=float),
            eq0_res=np.zeros((0,), dtype=float),
            act_lo0=np.zeros((nred,), dtype=bool),
            act_hi0=np.zeros((nred,), dtype=bool),
            lambda0_eq=np.zeros((0,), dtype=float),
            S_red=0.25 * np.ones((nred,), dtype=float),
            S_lam_eq=np.zeros((0,), dtype=float),
            c_red=np.ones((nred,), dtype=float),
            eq_data=eq_data,
            funcs=funcs,
            coeffs={},
            bcs_now=[],
            lo_red=lo_red,
            hi_red=hi_red,
            eq_prepare_callback=eq_prepare,
        )


def test_internal_pdas_vi_line_search_freezes_trial_model_when_guarded(monkeypatch) -> None:
    solver, _, alpha_k, _, phi_k, _ = _build_two_field_vi_solver(
        vi_params=VIParameters(
            c=1.0,
            project_each_iteration=True,
        )
    )

    funcs = [alpha_k, phi_k]
    lo_red, hi_red = solver._bounds_reduced()
    nred = int(lo_red.size)
    state = {"x": np.zeros((nred,), dtype=float), "proj_eq_prepare_is_none": False}
    solver._vi_working_set_guard_active = np.ones((nred,), dtype=np.int8)

    monkeypatch.setattr(solver.restrictor, "expand_vec", lambda v: np.asarray(v, dtype=float).copy())
    monkeypatch.setattr(
        solver.dh,
        "add_to_functions",
        lambda step, funcs_in: state.__setitem__("x", state["x"] + np.asarray(step, dtype=float)),
    )
    monkeypatch.setattr(solver.dh, "apply_bcs", lambda bcs, *funcs_in: None)
    monkeypatch.setattr(solver, "_bounds_reduced", lambda: (_ for _ in ()).throw(RuntimeError("bounds_recomputed")))
    monkeypatch.setattr(
        solver,
        "_project_funcs_to_bounds",
        lambda funcs_in, bcs_in, lo_in, hi_in, *, eq_data=None, eq_prepare_callback=None: state.__setitem__(
            "proj_eq_prepare_is_none",
            eq_prepare_callback is None,
        ),
    )
    monkeypatch.setattr(
        solver,
        "_assemble_system_reduced",
        lambda coeffs, need_matrix=False: (None, 1.0e-2 * (1.0 - state["x"])),
    )
    monkeypatch.setattr(solver, "_pack_reduced_iterate", lambda funcs_in: state["x"].copy())
    monkeypatch.setattr(solver, "_vi_stationarity_residual", lambda R_red, eq_data, lam_eq: np.asarray(R_red, dtype=float))
    monkeypatch.setattr(
        solver,
        "_pdas_residual_and_sets",
        lambda x_red, R_red, lo_red_in, hi_red_in, c_red=None: (
            np.asarray(R_red, dtype=float).copy(),
            np.zeros(x_red.shape, dtype=bool),
            np.zeros(x_red.shape, dtype=bool),
        ),
    )
    monkeypatch.setattr(solver, "_vi_equality_residual", lambda x_red, eq_data: np.zeros((0,), dtype=float))

    eq_prepare = lambda: (_ for _ in ()).throw(RuntimeError("eq_prepare_called"))
    eq_data = _PreparedLinearEqualities(
        names=tuple(),
        field_names=tuple(),
        B_red=np.zeros((0, nred), dtype=float),
        b_eff=np.zeros((0,), dtype=float),
        project_feasible=tuple(),
    )

    step, _ = solver._line_search_vi_reduced(
        x0_red=np.zeros((nred,), dtype=float),
        R0_red=1.0e-2 * np.ones((nred,), dtype=float),
        G0=1.0e-2 * np.ones((nred,), dtype=float),
        eq0_res=np.zeros((0,), dtype=float),
        act_lo0=np.zeros((nred,), dtype=bool),
        act_hi0=np.zeros((nred,), dtype=bool),
        lambda0_eq=np.zeros((0,), dtype=float),
        S_red=np.ones((nred,), dtype=float),
        S_lam_eq=np.zeros((0,), dtype=float),
        c_red=np.ones((nred,), dtype=float),
        eq_data=eq_data,
        funcs=funcs,
        coeffs={},
        bcs_now=[],
        lo_red=lo_red,
        hi_red=hi_red,
        eq_prepare_callback=eq_prepare,
    )

    assert np.allclose(step, np.ones((nred,), dtype=float))
    assert state["proj_eq_prepare_is_none"] is True


def test_internal_pdas_vi_line_search_keeps_active_step_policy_with_equalities(monkeypatch) -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            c=1.0,
            project_each_iteration=False,
            equality_active_step_ginf_threshold=5.0e-2,
        )
    )

    lo_red, hi_red = solver._bounds_reduced()
    nred = int(lo_red.size)
    state = {"x": np.zeros((nred,), dtype=float)}
    active_step_calls: list[float] = []

    def _residual_for(x_red: np.ndarray) -> np.ndarray:
        return 1.0e-2 * (1.0 - np.asarray(x_red, dtype=float))

    def _active_step_policy(*, alpha, S_red, active_mask, strong_active_mask, active_delta_ref):
        active_step_calls.append(float(alpha))
        return float(alpha) * np.asarray(S_red, dtype=float), False

    monkeypatch.setattr(solver, "_vi_apply_active_step_policy", _active_step_policy)
    monkeypatch.setattr(solver.restrictor, "expand_vec", lambda v: np.asarray(v, dtype=float).copy())
    monkeypatch.setattr(
        solver.dh,
        "add_to_functions",
        lambda step, funcs: state.__setitem__("x", state["x"] + np.asarray(step, dtype=float)),
    )
    monkeypatch.setattr(solver.dh, "apply_bcs", lambda bcs, *funcs: None)
    monkeypatch.setattr(solver, "_assemble_system_reduced", lambda coeffs, need_matrix=False: (None, _residual_for(state["x"])))
    monkeypatch.setattr(solver, "_pack_reduced_iterate", lambda funcs: state["x"].copy())
    monkeypatch.setattr(solver, "_vi_stationarity_residual", lambda R_red, eq_data, lam_eq: np.asarray(R_red, dtype=float))
    monkeypatch.setattr(
        solver,
        "_pdas_residual_and_sets",
        lambda x_red, R_red, lo_red_in, hi_red_in, c_red=None: (
            np.asarray(R_red, dtype=float).copy(),
            np.zeros(x_red.shape, dtype=bool),
            np.zeros(x_red.shape, dtype=bool),
        ),
    )
    monkeypatch.setattr(solver, "_vi_equality_residual", lambda x_red, eq_data: np.zeros((1,), dtype=float))

    x0_red = np.zeros((nred,), dtype=float)
    R0_red = _residual_for(x0_red)
    G0 = R0_red.copy()
    eq0_res = np.zeros((1,), dtype=float)
    act_lo0 = np.zeros((nred,), dtype=bool)
    act_hi0 = np.zeros((nred,), dtype=bool)
    lambda0_eq = np.zeros((1,), dtype=float)
    c_red = np.ones((nred,), dtype=float)
    eq_data = _PreparedLinearEqualities(
        names=("alpha_mass",),
        field_names=("u",),
        B_red=np.ones((1, nred), dtype=float),
        b_eff=np.zeros((1,), dtype=float),
        project_feasible=tuple(),
    )

    step, lam_try = solver._line_search_vi_reduced(
        x0_red=x0_red,
        R0_red=R0_red,
        G0=G0,
        eq0_res=eq0_res,
        act_lo0=act_lo0,
        act_hi0=act_hi0,
        lambda0_eq=lambda0_eq,
        S_red=np.ones((nred,), dtype=float),
        S_lam_eq=np.zeros((1,), dtype=float),
        c_red=c_red,
        eq_data=eq_data,
        funcs=[],
        coeffs={},
        bcs_now=[],
        lo_red=lo_red,
        hi_red=hi_red,
    )

    assert active_step_calls
    assert active_step_calls[0] == pytest.approx(1.0)
    assert np.allclose(step, np.ones((nred,), dtype=float))
    assert np.allclose(lam_try, np.zeros((1,), dtype=float))


def test_internal_pdas_vi_line_search_keeps_uniform_step_for_large_equality_residual(monkeypatch) -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            c=1.0,
            project_each_iteration=False,
            equality_active_step_ginf_threshold=5.0e-2,
        )
    )

    lo_red, hi_red = solver._bounds_reduced()
    nred = int(lo_red.size)
    state = {"x": np.zeros((nred,), dtype=float)}
    active_step_calls: list[float] = []

    def _residual_for(x_red: np.ndarray) -> np.ndarray:
        return 1.0 - np.asarray(x_red, dtype=float)

    def _active_step_policy(*, alpha, S_red, active_mask, strong_active_mask, active_delta_ref):
        active_step_calls.append(float(alpha))
        return float(alpha) * np.asarray(S_red, dtype=float), False

    monkeypatch.setattr(solver, "_vi_apply_active_step_policy", _active_step_policy)
    monkeypatch.setattr(solver.restrictor, "expand_vec", lambda v: np.asarray(v, dtype=float).copy())
    monkeypatch.setattr(
        solver.dh,
        "add_to_functions",
        lambda step, funcs: state.__setitem__("x", state["x"] + np.asarray(step, dtype=float)),
    )
    monkeypatch.setattr(solver.dh, "apply_bcs", lambda bcs, *funcs: None)
    monkeypatch.setattr(solver, "_assemble_system_reduced", lambda coeffs, need_matrix=False: (None, _residual_for(state["x"])))
    monkeypatch.setattr(solver, "_pack_reduced_iterate", lambda funcs: state["x"].copy())
    monkeypatch.setattr(solver, "_vi_stationarity_residual", lambda R_red, eq_data, lam_eq: np.asarray(R_red, dtype=float))
    monkeypatch.setattr(
        solver,
        "_pdas_residual_and_sets",
        lambda x_red, R_red, lo_red_in, hi_red_in, c_red=None: (
            np.asarray(R_red, dtype=float).copy(),
            np.zeros(x_red.shape, dtype=bool),
            np.zeros(x_red.shape, dtype=bool),
        ),
    )
    monkeypatch.setattr(solver, "_vi_equality_residual", lambda x_red, eq_data: np.zeros((1,), dtype=float))

    x0_red = np.zeros((nred,), dtype=float)
    R0_red = _residual_for(x0_red)
    G0 = R0_red.copy()
    eq0_res = np.zeros((1,), dtype=float)
    act_lo0 = np.zeros((nred,), dtype=bool)
    act_hi0 = np.zeros((nred,), dtype=bool)
    lambda0_eq = np.zeros((1,), dtype=float)
    c_red = np.ones((nred,), dtype=float)
    eq_data = _PreparedLinearEqualities(
        names=("alpha_mass",),
        field_names=("u",),
        B_red=np.ones((1, nred), dtype=float),
        b_eff=np.zeros((1,), dtype=float),
        project_feasible=tuple(),
    )

    step, lam_try = solver._line_search_vi_reduced(
        x0_red=x0_red,
        R0_red=R0_red,
        G0=G0,
        eq0_res=eq0_res,
        act_lo0=act_lo0,
        act_hi0=act_hi0,
        lambda0_eq=lambda0_eq,
        S_red=np.ones((nred,), dtype=float),
        S_lam_eq=np.zeros((1,), dtype=float),
        c_red=c_red,
        eq_data=eq_data,
        funcs=[],
        coeffs={},
        bcs_now=[],
        lo_red=lo_red,
        hi_red=hi_red,
    )

    assert not active_step_calls
    assert np.allclose(step, np.ones((nred,), dtype=float))
    assert np.allclose(lam_try, np.zeros((1,), dtype=float))


def test_internal_pdas_vi_line_search_disables_active_step_policy_on_fixed_ptc_manifold(
    monkeypatch,
) -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            c=1.0,
            project_each_iteration=False,
            ptc_recovery=True,
        )
    )

    lo_red, hi_red = solver._bounds_reduced()
    nred = int(lo_red.size)
    state = {"x": np.zeros((nred,), dtype=float)}
    active_step_calls: list[float] = []
    solver._vi_ptc_sigma_current = 5.0e-2

    def _residual_for(x_red: np.ndarray) -> np.ndarray:
        return 1.0e-2 * (1.0 - np.asarray(x_red, dtype=float))

    def _active_step_policy(*, alpha, S_red, active_mask, strong_active_mask, active_delta_ref):
        active_step_calls.append(float(alpha))
        return float(alpha) * np.asarray(S_red, dtype=float), False

    monkeypatch.setattr(solver, "_vi_apply_active_step_policy", _active_step_policy)
    monkeypatch.setattr(solver.restrictor, "expand_vec", lambda v: np.asarray(v, dtype=float).copy())
    monkeypatch.setattr(
        solver.dh,
        "add_to_functions",
        lambda step, funcs: state.__setitem__("x", state["x"] + np.asarray(step, dtype=float)),
    )
    monkeypatch.setattr(solver.dh, "apply_bcs", lambda bcs, *funcs: None)
    monkeypatch.setattr(solver, "_assemble_system_reduced", lambda coeffs, need_matrix=False: (None, _residual_for(state["x"])))
    monkeypatch.setattr(solver, "_pack_reduced_iterate", lambda funcs: state["x"].copy())
    monkeypatch.setattr(solver, "_vi_stationarity_residual", lambda R_red, eq_data, lam_eq: np.asarray(R_red, dtype=float))
    monkeypatch.setattr(
        solver,
        "_vi_equality_residual",
        lambda x_red, eq_data: np.zeros((0,), dtype=float),
    )

    x0_red = np.zeros((nred,), dtype=float)
    R0_red = _residual_for(x0_red)
    G0 = R0_red.copy()
    eq0_res = np.zeros((0,), dtype=float)
    act_lo0 = np.zeros((nred,), dtype=bool)
    act_hi0 = np.zeros((nred,), dtype=bool)
    lambda0_eq = np.zeros((0,), dtype=float)
    c_red = np.ones((nred,), dtype=float)
    eq_data = _PreparedLinearEqualities(
        names=tuple(),
        field_names=tuple(),
        B_red=np.zeros((0, nred), dtype=float),
        b_eff=np.zeros((0,), dtype=float),
        project_feasible=tuple(),
    )

    step, lam_try = solver._line_search_vi_reduced(
        x0_red=x0_red,
        R0_red=R0_red,
        G0=G0,
        eq0_res=eq0_res,
        act_lo0=act_lo0,
        act_hi0=act_hi0,
        lambda0_eq=lambda0_eq,
        S_red=np.ones((nred,), dtype=float),
        S_lam_eq=np.zeros((0,), dtype=float),
        c_red=c_red,
        eq_data=eq_data,
        funcs=[],
        coeffs={},
        bcs_now=[],
        lo_red=lo_red,
        hi_red=hi_red,
    )

    assert not active_step_calls
    assert np.allclose(step, np.ones((nred,), dtype=float))
    assert np.allclose(lam_try, np.zeros((0,), dtype=float))


def test_internal_pdas_vi_line_search_accepts_nonmonotone_step_once_active_set_is_stable(monkeypatch) -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            c=1.0,
            project_each_iteration=False,
            line_search_nonmonotone_window=3,
            line_search_nonmonotone_active_stable_iters=1,
            line_search_nonmonotone_ginf_trigger=2.0,
            line_search_nonmonotone_gap_ratio=1.0,
            line_search_nonmonotone_eq_abs=1.0e-12,
        )
    )

    lo_red, hi_red = solver._bounds_reduced()
    nred = int(lo_red.size)
    state = {"x": np.zeros((nred,), dtype=float)}

    monkeypatch.setattr(solver.restrictor, "expand_vec", lambda v: np.asarray(v, dtype=float).copy())
    monkeypatch.setattr(
        solver.dh,
        "add_to_functions",
        lambda step, funcs: state.__setitem__("x", state["x"] + np.asarray(step, dtype=float)),
    )
    monkeypatch.setattr(solver.dh, "apply_bcs", lambda bcs, *funcs: None)
    monkeypatch.setattr(
        solver,
        "_assemble_system_reduced",
        lambda coeffs, need_matrix=False: (None, np.zeros((nred,), dtype=float)),
    )
    monkeypatch.setattr(solver, "_pack_reduced_iterate", lambda funcs: state["x"].copy())
    monkeypatch.setattr(
        solver,
        "_vi_stationarity_residual",
        lambda R_red, eq_data, lam_eq: np.zeros((nred,), dtype=float),
    )
    monkeypatch.setattr(
        solver,
        "_pdas_residual_and_sets",
        lambda x_red, R_red, lo_red_in, hi_red_in, c_red=None: (
            np.zeros((nred,), dtype=float),
            np.zeros((nred,), dtype=bool),
            np.zeros((nred,), dtype=bool),
        ),
    )
    monkeypatch.setattr(solver, "_vi_equality_residual", lambda x_red, eq_data: np.zeros((0,), dtype=float))

    def _metrics(x_red, stat_red, lo_red_in, hi_red_in, act_lo, act_hi, G_red, eq_res):
        alpha = float(np.max(np.asarray(x_red, dtype=float))) if np.asarray(x_red).size else 0.0
        if alpha <= 1.0e-14:
            return {
                "G_inf": 1.0,
                "G_half": 0.5,
                "inactive_res_inf": 1.0,
                "active_gap_inf": 0.0,
                "equality_inf": 0.0,
                "inactive_res_l2_sq": 1.0,
                "active_gap_l2_sq": 0.0,
                "equality_l2_sq": 0.0,
                "merit": 1.0,
            }
        return {
            "G_inf": 8.0e-1,
            "G_half": 0.32,
            "inactive_res_inf": 8.0e-1,
            "active_gap_inf": 0.0,
            "equality_inf": 0.0,
            "inactive_res_l2_sq": 8.0e-1,
            "active_gap_l2_sq": 0.0,
            "equality_l2_sq": 0.0,
            "merit": 1.05,
        }

    monkeypatch.setattr(solver, "_vi_metrics", _metrics)

    x0_red = np.zeros((nred,), dtype=float)
    R0_red = np.zeros((nred,), dtype=float)
    G0 = np.zeros((nred,), dtype=float)
    eq0_res = np.zeros((0,), dtype=float)
    act_lo0 = np.zeros((nred,), dtype=bool)
    act_hi0 = np.zeros((nred,), dtype=bool)
    lambda0_eq = np.zeros((0,), dtype=float)
    c_red = np.ones((nred,), dtype=float)
    eq_data = _PreparedLinearEqualities(
        names=tuple(),
        field_names=tuple(),
        B_red=np.zeros((0, nred), dtype=float),
        b_eff=np.zeros((0,), dtype=float),
        project_feasible=tuple(),
    )

    step, lam_try = solver._line_search_vi_reduced(
        x0_red=x0_red,
        R0_red=R0_red,
        G0=G0,
        eq0_res=eq0_res,
        act_lo0=act_lo0,
        act_hi0=act_hi0,
        lambda0_eq=lambda0_eq,
        S_red=np.ones((nred,), dtype=float),
        S_lam_eq=np.zeros((0,), dtype=float),
        c_red=c_red,
        eq_data=eq_data,
        funcs=[],
        coeffs={},
        bcs_now=[],
        lo_red=lo_red,
        hi_red=hi_red,
        active_stable_count=1,
        merit_history=[1.2, 1.1, 1.0],
    )

    assert np.allclose(step, np.ones((nred,), dtype=float))
    assert np.allclose(lam_try, np.zeros((0,), dtype=float))


def test_internal_pdas_vi_line_search_local_merit_mode_bypasses_filter_once_active_set_is_stable(
    monkeypatch,
) -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            c=1.0,
            project_each_iteration=False,
            line_search_nonmonotone_window=3,
            line_search_nonmonotone_active_stable_iters=1,
            line_search_nonmonotone_ginf_trigger=2.0,
            line_search_nonmonotone_gap_ratio=1.0,
            line_search_nonmonotone_eq_abs=1.0e-12,
            line_search_nonmonotone_disable_filter=True,
        )
    )

    lo_red, hi_red = solver._bounds_reduced()
    nred = int(lo_red.size)
    state = {"x": np.zeros((nred,), dtype=float)}

    monkeypatch.setattr(solver.restrictor, "expand_vec", lambda v: np.asarray(v, dtype=float).copy())
    monkeypatch.setattr(
        solver.dh,
        "add_to_functions",
        lambda step, funcs: state.__setitem__("x", state["x"] + np.asarray(step, dtype=float)),
    )
    monkeypatch.setattr(solver.dh, "apply_bcs", lambda bcs, *funcs: None)
    monkeypatch.setattr(
        solver,
        "_assemble_system_reduced",
        lambda coeffs, need_matrix=False: (None, np.zeros((nred,), dtype=float)),
    )
    monkeypatch.setattr(solver, "_pack_reduced_iterate", lambda funcs: state["x"].copy())
    monkeypatch.setattr(
        solver,
        "_vi_stationarity_residual",
        lambda R_red, eq_data, lam_eq: np.zeros((nred,), dtype=float),
    )
    monkeypatch.setattr(
        solver,
        "_pdas_residual_and_sets",
        lambda x_red, R_red, lo_red_in, hi_red_in, c_red=None: (
            np.zeros((nred,), dtype=float),
            np.zeros((nred,), dtype=bool),
            np.zeros((nred,), dtype=bool),
        ),
    )
    monkeypatch.setattr(solver, "_vi_equality_residual", lambda x_red, eq_data: np.zeros((0,), dtype=float))

    def _metrics(x_red, stat_red, lo_red_in, hi_red_in, act_lo, act_hi, G_red, eq_res):
        alpha = float(np.max(np.asarray(x_red, dtype=float))) if np.asarray(x_red).size else 0.0
        if alpha <= 1.0e-14:
            return {
                "G_inf": 1.0,
                "G_half": 0.5,
                "inactive_res_inf": 1.0,
                "active_gap_inf": 0.0,
                "equality_inf": 0.0,
                "inactive_res_l2_sq": 1.0,
                "active_gap_l2_sq": 0.0,
                "equality_l2_sq": 0.0,
                "merit": 1.0,
            }
        return {
            "G_inf": 8.0e-1,
            "G_half": 0.32,
            "inactive_res_inf": 1.8,
            "active_gap_inf": 6.0e-1,
            "equality_inf": 0.0,
            "inactive_res_l2_sq": 8.0e-1,
            "active_gap_l2_sq": 2.0e-2,
            "equality_l2_sq": 0.0,
            "merit": 8.5e-1,
        }

    monkeypatch.setattr(solver, "_vi_metrics", _metrics)

    x0_red = np.zeros((nred,), dtype=float)
    R0_red = np.zeros((nred,), dtype=float)
    G0 = np.zeros((nred,), dtype=float)
    eq0_res = np.zeros((0,), dtype=float)
    act_lo0 = np.zeros((nred,), dtype=bool)
    act_hi0 = np.zeros((nred,), dtype=bool)
    lambda0_eq = np.zeros((0,), dtype=float)
    c_red = np.ones((nred,), dtype=float)
    eq_data = _PreparedLinearEqualities(
        names=tuple(),
        field_names=tuple(),
        B_red=np.zeros((0, nred), dtype=float),
        b_eff=np.zeros((0,), dtype=float),
        project_feasible=tuple(),
    )

    step, lam_try = solver._line_search_vi_reduced(
        x0_red=x0_red,
        R0_red=R0_red,
        G0=G0,
        eq0_res=eq0_res,
        act_lo0=act_lo0,
        act_hi0=act_hi0,
        lambda0_eq=lambda0_eq,
        S_red=np.ones((nred,), dtype=float),
        S_lam_eq=np.zeros((0,), dtype=float),
        c_red=c_red,
        eq_data=eq_data,
        funcs=[],
        coeffs={},
        bcs_now=[],
        lo_red=lo_red,
        hi_red=hi_red,
        active_stable_count=1,
        merit_history=[1.0, 9.5e-1, 9.0e-1],
    )

    assert np.allclose(step, np.ones((nred,), dtype=float))
    assert np.allclose(lam_try, np.zeros((0,), dtype=float))


def test_internal_pdas_stall_regularization_skips_gap_free_stagnation() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            inactive_reg_lambda0=1.0,
            inactive_reg_lambda_max=1.0e3,
            inactive_reg_growth=5.0,
            inactive_reg_stall_alpha=0.25,
            inactive_reg_stall_merit_ratio=0.95,
        )
    )

    assert not solver._vi_should_grow_inactive_regularization(
        accepted_alpha=1.0e-4,
        merit_ratio=0.999,
        accepted_metrics={
            "inactive_res_inf": 8.14,
            "active_gap_inf": 1.0e-14,
        },
        current_lambda=1.0,
    )

    assert solver._vi_should_grow_inactive_regularization(
        accepted_alpha=1.0e-4,
        merit_ratio=0.999,
        accepted_metrics={
            "inactive_res_inf": 1.0e-3,
            "active_gap_inf": 5.0e-3,
        },
        current_lambda=1.0,
    )


def test_internal_pdas_working_set_guard_freezes_active_state_for_guarded_iteration() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            c=1.0,
            enter_tol=1.0e-6,
            leave_tol=1.0e-6,
            working_set_guard_after_affine=1,
        )
    )
    lo_red, hi_red = solver._bounds_reduced()
    x_red = lo_red.copy()

    solver._vi_arm_working_set_guard(np.ones(x_red.shape, dtype=np.int8))
    solver._vi_begin_working_set_guard_iteration()

    _, act_lo, act_hi = solver._pdas_residual_and_sets(
        x_red,
        np.full(x_red.shape, -1.0, dtype=float),
        lo_red,
        hi_red,
        c_red=np.ones(x_red.shape, dtype=float),
    )

    assert np.all(act_lo)
    assert not np.any(act_hi)
    assert solver._vi_working_set_guard_remaining == 0


def test_internal_pdas_working_set_guard_expires_after_configured_iterations() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            c=1.0,
            enter_tol=1.0e-6,
            leave_tol=1.0e-6,
            working_set_guard_after_affine=1,
        )
    )
    lo_red, hi_red = solver._bounds_reduced()
    x_red = lo_red.copy()

    solver._vi_arm_working_set_guard(np.ones(x_red.shape, dtype=np.int8))
    solver._vi_begin_working_set_guard_iteration()
    solver._pdas_residual_and_sets(
        x_red,
        np.full(x_red.shape, -1.0, dtype=float),
        lo_red,
        hi_red,
        c_red=np.ones(x_red.shape, dtype=float),
    )

    solver._vi_begin_working_set_guard_iteration()
    _, act_lo, act_hi = solver._pdas_residual_and_sets(
        x_red,
        np.full(x_red.shape, -1.0, dtype=float),
        lo_red,
        hi_red,
        c_red=np.ones(x_red.shape, dtype=float),
    )

    assert not np.any(act_lo)
    assert not np.any(act_hi)


def test_internal_pdas_vi_line_search_restores_full_guard_state_after_trial(monkeypatch) -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            c=1.0,
            project_each_iteration=False,
        )
    )
    lo_red, hi_red = solver._bounds_reduced()
    nred = int(lo_red.size)
    state = {"x": np.zeros((nred,), dtype=float)}

    solver._vi_prev_state = np.ones((nred,), dtype=np.int8)
    solver._vi_pending_state = -np.ones((nred,), dtype=np.int8)
    solver._vi_pending_count = np.arange(nred, dtype=np.int16)
    solver._vi_forced_state = np.ones((nred,), dtype=np.int8)
    solver._vi_working_set_guard_state = -np.ones((nred,), dtype=np.int8)
    solver._vi_working_set_guard_remaining = 2
    solver._vi_working_set_guard_active = np.ones((nred,), dtype=np.int8)
    snap = solver._vi_capture_classification_state()

    monkeypatch.setattr(solver.restrictor, "expand_vec", lambda v: np.asarray(v, dtype=float).copy())
    monkeypatch.setattr(
        solver.dh,
        "add_to_functions",
        lambda step, funcs: state.__setitem__("x", state["x"] + np.asarray(step, dtype=float)),
    )
    monkeypatch.setattr(solver.dh, "apply_bcs", lambda bcs, *funcs: None)
    monkeypatch.setattr(
        solver,
        "_assemble_system_reduced",
        lambda coeffs, need_matrix=False: (None, np.zeros((nred,), dtype=float)),
    )
    monkeypatch.setattr(solver, "_pack_reduced_iterate", lambda funcs: state["x"].copy())
    monkeypatch.setattr(
        solver,
        "_vi_stationarity_residual",
        lambda R_red, eq_data, lam_eq: np.asarray(R_red, dtype=float),
    )

    def _consume_vi_state(x_red, R_red, lo_red_in, hi_red_in, c_red=None):
        solver._vi_prev_state = np.zeros((nred,), dtype=np.int8)
        solver._vi_pending_state = np.zeros((nred,), dtype=np.int8)
        solver._vi_pending_count = np.zeros((nred,), dtype=np.int16)
        solver._vi_forced_state = None
        solver._vi_working_set_guard_state = None
        solver._vi_working_set_guard_remaining = 0
        solver._vi_working_set_guard_active = None
        return (
            np.asarray(R_red, dtype=float).copy(),
            np.zeros((nred,), dtype=bool),
            np.zeros((nred,), dtype=bool),
        )

    monkeypatch.setattr(solver, "_pdas_residual_and_sets", _consume_vi_state)
    monkeypatch.setattr(solver, "_vi_equality_residual", lambda x_red, eq_data: np.zeros((0,), dtype=float))
    monkeypatch.setattr(
        solver,
        "_vi_metrics",
        lambda x_red, stat_red, lo_red_in, hi_red_in, act_lo, act_hi, G_red, eq_res: {
            "G_inf": 1.0 if np.max(np.asarray(x_red, dtype=float)) <= 1.0e-14 else 1.0e-1,
            "G_half": 5.0e-1 if np.max(np.asarray(x_red, dtype=float)) <= 1.0e-14 else 5.0e-2,
            "inactive_res_inf": 1.0 if np.max(np.asarray(x_red, dtype=float)) <= 1.0e-14 else 1.0e-1,
            "active_gap_inf": 0.0,
            "equality_inf": 0.0,
            "inactive_res_l2_sq": 1.0 if np.max(np.asarray(x_red, dtype=float)) <= 1.0e-14 else 1.0e-2,
            "active_gap_l2_sq": 0.0,
            "equality_l2_sq": 0.0,
            "merit": 1.0 if np.max(np.asarray(x_red, dtype=float)) <= 1.0e-14 else 1.0e-1,
        },
    )

    eq_data = _PreparedLinearEqualities(
        names=tuple(),
        field_names=tuple(),
        B_red=np.zeros((0, nred), dtype=float),
        b_eff=np.zeros((0,), dtype=float),
        project_feasible=tuple(),
    )

    solver._line_search_vi_reduced(
        x0_red=np.zeros((nred,), dtype=float),
        R0_red=np.zeros((nred,), dtype=float),
        G0=np.zeros((nred,), dtype=float),
        eq0_res=np.zeros((0,), dtype=float),
        act_lo0=np.zeros((nred,), dtype=bool),
        act_hi0=np.zeros((nred,), dtype=bool),
        lambda0_eq=np.zeros((0,), dtype=float),
        S_red=np.ones((nred,), dtype=float),
        S_lam_eq=np.zeros((0,), dtype=float),
        c_red=np.ones((nred,), dtype=float),
        eq_data=eq_data,
        funcs=[],
        coeffs={},
        bcs_now=[],
        lo_red=lo_red,
        hi_red=hi_red,
    )

    restored = solver._vi_capture_classification_state()
    assert np.array_equal(restored["prev_state"], snap["prev_state"])
    assert np.array_equal(restored["pending_state"], snap["pending_state"])
    assert np.array_equal(restored["pending_count"], snap["pending_count"])
    assert np.array_equal(restored["forced_state"], snap["forced_state"])
    assert np.array_equal(restored["guard_state"], snap["guard_state"])
    assert restored["guard_remaining"] == snap["guard_remaining"]
    assert np.array_equal(restored["guard_active"], snap["guard_active"])


def test_internal_pdas_guard_state_after_affine_rescue_only_adds_blocking_bounds() -> None:
    solver = _build_scalar_vi_solver(vi_params=VIParameters())
    state_base = np.array([1, 0, 0, 0], dtype=np.int8)
    state_affine = np.array([1, 1, -1, -1], dtype=np.int8)
    x_affine = np.array([0.0, 1.0e-10, 0.75, 1.0], dtype=float)
    lo_red = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    hi_red = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

    guard_state = solver._vi_guard_state_after_affine_rescue(
        state_base=state_base,
        state_affine=state_affine,
        x_affine=x_affine,
        lo_red=lo_red,
        hi_red=hi_red,
    )

    assert np.array_equal(guard_state, np.array([1, 1, 0, -1], dtype=np.int8))


def test_internal_pdas_rebuild_carried_state_uses_endpoint_classification() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(c=1.0, enter_tol=1.0e-6, leave_tol=1.0e-6)
    )
    lo_red, hi_red = solver._bounds_reduced()
    x_red = lo_red.copy()
    stat_red = np.ones(x_red.shape, dtype=float)
    c_red = np.ones(x_red.shape, dtype=float)

    state = solver._vi_rebuild_carried_state_from_endpoint(
        x_red=x_red,
        stationarity_red=stat_red,
        lo_red=lo_red,
        hi_red=hi_red,
        c_red=c_red,
        state_base=np.zeros(x_red.shape, dtype=np.int8),
    )

    assert np.all(np.asarray(state, dtype=np.int8) == 1)


def test_internal_pdas_predicted_linear_sets_use_physical_equality_coupling(monkeypatch) -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(
            c=1.0,
            project_each_iteration=False,
            equality_active_step_ginf_threshold=float("inf"),
        )
    )

    lo_red, hi_red = solver._bounds_reduced()
    nred = int(lo_red.size)
    state = {"x": np.zeros((nred,), dtype=float)}
    captured: dict[str, np.ndarray] = {}

    monkeypatch.setattr(solver.restrictor, "expand_vec", lambda v: np.asarray(v, dtype=float).copy())
    monkeypatch.setattr(
        solver.dh,
        "add_to_functions",
        lambda step, funcs: state.__setitem__("x", state["x"] + np.asarray(step, dtype=float)),
    )
    monkeypatch.setattr(solver.dh, "apply_bcs", lambda bcs, *funcs: None)
    monkeypatch.setattr(
        solver,
        "_assemble_system_reduced",
        lambda coeffs, need_matrix=False: (None, np.zeros((nred,), dtype=float)),
    )
    monkeypatch.setattr(solver, "_pack_reduced_iterate", lambda funcs: state["x"].copy())
    monkeypatch.setattr(
        solver,
        "_vi_stationarity_residual",
        lambda R_red, eq_data, lam_eq, row_scale_red=None: np.asarray(R_red, dtype=float),
    )
    monkeypatch.setattr(
        solver,
        "_pdas_residual_and_sets",
        lambda x_red, R_red, lo_red_in, hi_red_in, c_red=None: (
            np.asarray(R_red, dtype=float).copy(),
            np.zeros(x_red.shape, dtype=bool),
            np.zeros(x_red.shape, dtype=bool),
        ),
    )
    monkeypatch.setattr(solver, "_vi_equality_residual", lambda x_red, eq_data: np.zeros((1,), dtype=float))
    monkeypatch.setattr(
        solver,
        "_vi_metrics",
        lambda x_red, stat_red, lo_red_in, hi_red_in, act_lo, act_hi, G_red, eq_res: {
            "G_inf": 1.0 if np.max(np.asarray(x_red, dtype=float)) <= 1.0e-14 else 1.0e-1,
            "G_half": 5.0e-1 if np.max(np.asarray(x_red, dtype=float)) <= 1.0e-14 else 5.0e-2,
            "inactive_res_inf": 1.0 if np.max(np.asarray(x_red, dtype=float)) <= 1.0e-14 else 1.0e-1,
            "active_gap_inf": 0.0,
            "equality_inf": 0.0,
            "inactive_res_l2_sq": 1.0 if np.max(np.asarray(x_red, dtype=float)) <= 1.0e-14 else 1.0e-2,
            "active_gap_l2_sq": 0.0,
            "equality_l2_sq": 0.0,
            "merit": 1.0 if np.max(np.asarray(x_red, dtype=float)) <= 1.0e-14 else 1.0e-1,
        },
    )

    def _capture_predict_sets(*, x_red, stationarity_red, lo_red, hi_red, c_red, prev_state=None):
        captured["stationarity_red"] = np.asarray(stationarity_red, dtype=float).copy()
        return np.zeros(x_red.shape, dtype=bool), np.zeros(x_red.shape, dtype=bool)

    monkeypatch.setattr(solver, "_vi_predict_sets_from_state", _capture_predict_sets)

    eq_data = _PreparedLinearEqualities(
        names=("alpha_mass",),
        field_names=("u",),
        B_red=np.ones((1, nred), dtype=float),
        b_eff=np.zeros((1,), dtype=float),
        project_feasible=tuple(),
    )

    solver._line_search_vi_reduced(
        A_red=sp.csr_matrix((nred, nred), dtype=float),
        row_scale_red=np.full((nred,), 7.0, dtype=float),
        x0_red=np.zeros((nred,), dtype=float),
        R0_red=np.zeros((nred,), dtype=float),
        G0=np.zeros((nred,), dtype=float),
        eq0_res=np.zeros((1,), dtype=float),
        act_lo0=np.zeros((nred,), dtype=bool),
        act_hi0=np.zeros((nred,), dtype=bool),
        lambda0_eq=np.zeros((1,), dtype=float),
        S_red=np.ones((nred,), dtype=float),
        S_lam_eq=np.ones((1,), dtype=float),
        c_red=np.ones((nred,), dtype=float),
        eq_data=eq_data,
        funcs=[],
        coeffs={},
        bcs_now=[],
        lo_red=lo_red,
        hi_red=hi_red,
    )

    assert "stationarity_red" in captured
    assert np.allclose(captured["stationarity_red"], np.ones((nred,), dtype=float))


def test_internal_pdas_affine_fixed_working_set_solve_keeps_fixed_active_bound() -> None:
    solver = _build_scalar_vi_solver(
        vi_params=VIParameters(affine_cycle_fallback_max_it=4)
    )
    nred = int(solver.active_dofs.size)
    A_red = sp.eye(nred, format="csr")
    lo_red, hi_red = solver._bounds_reduced()
    x0_red = np.full((nred,), 0.25, dtype=float)
    x0_red[0] = 0.0
    rhs_aff_red = np.full((nred,), -0.5, dtype=float)
    eq_data = _PreparedLinearEqualities(
        names=tuple(),
        field_names=tuple(),
        B_red=np.zeros((0, nred), dtype=float),
        b_eff=np.zeros((0,), dtype=float),
        project_feasible=tuple(),
    )
    act_lo = np.zeros((nred,), dtype=bool)
    act_lo[0] = True
    act_hi = np.zeros((nred,), dtype=bool)

    result = solver._vi_affine_fixed_working_set_solve(
        A_red=A_red,
        rhs_aff_red=rhs_aff_red,
        x0_red=x0_red,
        lambda0_eq=np.zeros((0,), dtype=float),
        lo_red=lo_red,
        hi_red=hi_red,
        act_lo_fixed=act_lo,
        act_hi_fixed=act_hi,
        eq_data=eq_data,
        tol=1.0e-10,
    )

    assert result is not None
    y_new, lam_new, metrics_new = result
    assert lam_new.size == 0
    assert y_new[0] == pytest.approx(0.0, abs=1.0e-12)
    assert float(metrics_new["G_inf"]) < 1.0


def test_direct_solve_equilibration_modes_auto_enables_col_and_ruiz(monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_DIRECT_SOLVE_EQUILIBRATION", "auto")
    assert NewtonSolver._direct_solve_equilibration_modes() == ("col", "ruiz")

    monkeypatch.setenv("PYCUTFEM_DIRECT_SOLVE_EQUILIBRATION", "off")
    assert NewtonSolver._direct_solve_equilibration_modes() == tuple()


def test_direct_solve_ruiz_scaling_returns_finite_positive_scales() -> None:
    A = sp.csr_matrix(
        np.array(
            [
                [1.0e8, 1.0, 0.0],
                [1.0e-8, 1.0, 2.0],
                [0.0, 3.0, 4.0e-4],
            ],
            dtype=float,
        )
    )
    row_scale, col_scale = NewtonSolver._direct_solve_ruiz_scaling(A, iters=3)
    assert row_scale.shape == (3,)
    assert col_scale.shape == (3,)
    assert np.all(np.isfinite(row_scale))
    assert np.all(np.isfinite(col_scale))
    assert np.all(row_scale > 0.0)
    assert np.all(col_scale > 0.0)


def test_reduced_system_ruiz_scaling_improves_condition_and_preserves_step() -> None:
    solver, *_ = _build_two_field_newton_solver()
    solver.set_reduced_system_scaling(
        equation_row_scaling=True,
        variable_column_scaling=True,
        mode="ruiz",
        ruiz_iters=6,
    )

    A = sp.csr_matrix(
        np.array(
            [
                [1.0e8, 1.0, 0.0, 0.0],
                [1.0e-6, 2.0, 0.0, 0.0],
                [0.0, 0.0, 5.0e-5, 1.0],
                [0.0, 0.0, 1.0, 3.0],
            ],
            dtype=float,
        )
    )
    R = np.array([1.0, -2.0, 0.5, -1.0], dtype=float)

    A_scaled, R_scaled, row_scale, col_scale = solver._apply_reduced_system_scaling(A, R)
    step_scaled = np.linalg.solve(A_scaled.toarray(), -R_scaled)
    step = col_scale * step_scaled
    step_ref = np.linalg.solve(A.toarray(), -R)

    s_raw = np.linalg.svd(A.toarray(), compute_uv=False)
    s_scaled = np.linalg.svd(A_scaled.toarray(), compute_uv=False)
    cond_raw = float(s_raw[0] / s_raw[-1])
    cond_scaled = float(s_scaled[0] / s_scaled[-1])

    assert cond_scaled < cond_raw
    assert cond_scaled < 1.0e6
    assert np.allclose(step, step_ref, rtol=1.0e-10, atol=1.0e-12)
    assert np.all(np.isfinite(row_scale))
    assert np.all(np.isfinite(col_scale))
    assert np.all(row_scale > 0.0)
    assert np.all(col_scale > 0.0)


def test_direct_solve_quality_accepts_usable_solution_when_globalized(monkeypatch) -> None:
    solver, *_ = _build_two_field_newton_solver()
    solver.np.globalization = "line_search_then_trust"
    monkeypatch.setenv("PYCUTFEM_DIRECT_SOLVE_CHECK_ATOL", "1e-10")
    monkeypatch.setenv("PYCUTFEM_DIRECT_SOLVE_CHECK_RTOL", "1e-7")
    monkeypatch.setenv("PYCUTFEM_DIRECT_SOLVE_USABLE_ATOL", "1e-8")
    monkeypatch.setenv("PYCUTFEM_DIRECT_SOLVE_USABLE_RTOL", "1e-5")

    A = sp.eye(2, format="csr")
    rhs = np.array([1.0, 1.0], dtype=float)
    sol = np.array([1.0 + 5.0e-6, 1.0], dtype=float)

    assert solver._direct_solve_quality_ok(A, rhs, sol, solver_name="test")


def test_trust_region_dogleg_step_respects_radius() -> None:
    solver, *_ = _build_two_field_newton_solver()
    A = sp.csr_matrix(np.array([[2.0, 0.0], [0.0, 1.0]], dtype=float))
    R = np.array([1.0, 1.0], dtype=float)
    metric = np.ones((2,), dtype=float)
    newton_step = np.array([-0.5, -1.0], dtype=float)

    step, boundary = solver._trust_region_dogleg_step(A, R, newton_step, delta=0.25, metric_diag=metric)

    assert boundary
    assert float(np.linalg.norm(step)) <= 0.25 + 1.0e-12
