import numpy as np
import pytest
import scipy.sparse as sp

from examples.biofilms.benchmarks.seboldt.paper1_benchmark7_seboldt import (
    _alpha_cutoffs_for_distance_multiple,
    _alpha_equilibrium,
    _bind_solver_final_form_internal_zero_dirichlet,
    _eval_scalar_at_point,
    _FINAL_FORM_INTERNAL_ZERO_FLUID_TAG,
    _FINAL_FORM_INTERNAL_ZERO_POROUS_TAG,
    _build_bcs,
    _build_forms,
    _condition_balanced_kinematic_setup,
    _condition_balanced_field_scales,
    _condition_balanced_solid_cutoff_y,
    _condition_balanced_volume_setup,
    _create_problem,
    _refresh_solver_final_form_internal_zero_dirichlet,
    _refresh_solver_inactive_solid_interface_band,
    _reduced_field_scale_vector,
    _set_solver_active_fields_with_tracking,
    _solver_requested_active_dofs,
    _tag_final_form_internal_zero_dirichlet_dofs,
    _tag_inactive_solid_dofs_above_y,
    _tag_inactive_solid_dofs_outside_interface_band,
    _zero_tagged_internal_zero_dirichlet_state,
)
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, NewtonSolver
from pycutfem.ufl.expressions import Constant
from pycutfem.ufl.forms import BoundaryCondition


def _initialize_small_benchmark7_state(problem: dict[str, object], *, eps_alpha: float, phi_b: float) -> None:
    alpha_init = lambda x, y: _alpha_equilibrium(y, y_interface=1.0, eps_alpha=float(eps_alpha))
    problem["alpha_n"].set_values_from_function(lambda x, y: float(alpha_init(x, y)))
    problem["alpha_k"].nodal_values[:] = problem["alpha_n"].nodal_values[:]

    if problem["phi_n"] is not None:
        phi_init = np.clip(
            1.0 - (1.0 - float(phi_b)) * np.asarray(problem["alpha_n"].nodal_values, dtype=float),
            0.0,
            1.0,
        )
        problem["phi_n"].nodal_values[:] = phi_init
        problem["phi_k"].nodal_values[:] = phi_init

    problem["p_n"].set_values_from_function(lambda x, y: 0.15 + 0.05 * x - 0.03 * y)
    problem["p_k"].set_values_from_function(lambda x, y: 0.22 + 0.07 * x - 0.04 * y)
    problem["v_n"].set_values_from_function(lambda x, y: np.array([0.03 + 0.01 * x, -0.02 + 0.015 * y]))
    problem["v_k"].set_values_from_function(lambda x, y: np.array([0.05 + 0.015 * x, -0.03 + 0.020 * y]))
    problem["vS_n"].set_values_from_function(lambda x, y: np.array([0.01 + 0.01 * x, -0.01 + 0.005 * y]))
    problem["vS_k"].set_values_from_function(lambda x, y: np.array([0.02 + 0.012 * x, -0.015 + 0.008 * y]))
    problem["u_n"].set_values_from_function(lambda x, y: np.array([0.01 * x * (1.0 - x), -0.004 * y]))
    problem["u_k"].set_values_from_function(lambda x, y: np.array([0.015 * x * (1.0 - x), -0.006 * y]))
    if problem.get("lambda_drag_n") is not None:
        problem["lambda_drag_n"].set_values_from_function(lambda x, y: np.array([0.04 - 0.01 * x, 0.02 * y - 0.01]))
        problem["lambda_drag_k"].set_values_from_function(lambda x, y: np.array([0.05 - 0.015 * x, 0.03 * y - 0.012]))

    alpha_xy = np.asarray(problem["dh"].get_dof_coords("alpha"), dtype=float)
    problem["alpha_k"].nodal_values[:] += 0.02 * np.sin(2.0 * np.pi * alpha_xy[:, 0]) * np.sin(np.pi * alpha_xy[:, 1] / 1.5)
    if problem.get("mu_n") is not None:
        problem["mu_n"].nodal_values[:] = 0.05 * np.asarray(problem["alpha_n"].nodal_values, dtype=float)
    if problem.get("mu_k") is not None:
        problem["mu_k"].nodal_values[:] = 0.07 * np.asarray(problem["alpha_k"].nodal_values, dtype=float)

    if problem["phi_k"] is not None:
        phi_xy = np.asarray(problem["dh"].get_dof_coords("phi"), dtype=float)
        problem["phi_k"].nodal_values[:] = np.clip(
            np.asarray(problem["phi_k"].nodal_values, dtype=float) + 0.01 * np.cos(np.pi * phi_xy[:, 0]),
            0.0,
            1.0,
        )
        if problem.get("S_n") is not None:
            problem["S_n"].nodal_values[:] = 1.0
        if problem.get("S_k") is not None:
            problem["S_k"].nodal_values[:] = 1.0


def _expected_interface_band_inactive_solid_dofs(
    problem: dict[str, object],
    *,
    reference_y: float,
    band_halfwidth: float = 0.25,
    alpha_state_key: str = "alpha_k",
) -> tuple[str, dict[str, set[int]]]:
    dh = problem["dh"]
    alpha_func = problem[alpha_state_key]
    alpha_global = np.zeros((int(dh.total_dofs),), dtype=float)
    alpha_g = np.asarray(getattr(alpha_func, "_g_dofs", np.array([], dtype=int)), dtype=int).ravel()
    alpha_global[alpha_g] = np.asarray(alpha_func.nodal_values, dtype=float).ravel()
    alpha_xy = np.asarray(dh.get_dof_coords("alpha"), dtype=float)
    alpha_vals = np.asarray(alpha_func.nodal_values, dtype=float).ravel()
    tol = 1.0e-12 * max(1.0, abs(float(reference_y)))
    probe = np.asarray(alpha_vals[alpha_xy[:, 1] > float(reference_y) + tol], dtype=float)
    if probe.size == 0:
        probe = np.asarray(alpha_vals, dtype=float)
    phase = "high" if float(np.mean(probe)) >= 0.5 else "low"
    lo_thr = 0.5 - float(band_halfwidth)
    hi_thr = 0.5 + float(band_halfwidth)
    inactive_elements: set[int] = set()
    for eid, gds in enumerate(list(getattr(dh, "element_maps", {}).get("alpha", []) or [])):
        g_arr = np.asarray(gds, dtype=int).ravel()
        if g_arr.size == 0:
            continue
        nodal = np.asarray(alpha_global[g_arr], dtype=float)
        if phase == "high":
            if np.all(nodal >= hi_thr):
                inactive_elements.add(int(eid))
        else:
            if np.all(nodal <= lo_thr):
                inactive_elements.add(int(eid))

    expected: dict[str, set[int]] = {}
    for field in ("u_x", "u_y", "vS_x", "vS_y"):
        dof_to_elements: dict[int, set[int]] = {}
        for eid, gds in enumerate(list(getattr(dh, "element_maps", {}).get(field, []) or [])):
            for gd in np.asarray(gds, dtype=int).ravel().tolist():
                dof_to_elements.setdefault(int(gd), set()).add(int(eid))
        expected[field] = {
            int(gd) for gd, adj in dof_to_elements.items() if adj and adj.issubset(inactive_elements)
        }
    return phase, expected


def _expected_final_form_internal_zero_dofs(
    problem: dict[str, object],
    *,
    distance_multiple: float,
    alpha_state_key: str = "alpha_k",
    targets: tuple[str, ...] = ("vf", "vP", "vS", "u"),
) -> tuple[float, float, dict[str, set[int]], dict[str, set[int]]]:
    dh = problem["dh"]
    mesh = problem["mesh"]
    alpha_low, alpha_high = _alpha_cutoffs_for_distance_multiple(float(distance_multiple))
    alpha_func = problem[alpha_state_key]

    def _field_dofs_by_alpha(field: str, *, threshold: float, mode: str) -> set[int]:
        field_ids = np.asarray(dh.get_field_slice(str(field)), dtype=int).ravel()
        coords = np.asarray(dh.get_dof_coords(str(field)), dtype=float)
        selected: set[int] = set()
        for gd, xy in zip(field_ids.tolist(), coords.tolist()):
            alpha_val = float(_eval_scalar_at_point(dh, mesh, alpha_func, (float(xy[0]), float(xy[1]))))
            if mode == "high":
                if alpha_val >= float(threshold):
                    selected.add(int(gd))
            else:
                if alpha_val <= float(threshold):
                    selected.add(int(gd))
        return selected

    fluid_fields: list[str] = []
    porous_fields: list[str] = []
    selected = {str(name) for name in tuple(targets or tuple()) if str(name)}
    if "vf" in selected:
        fluid_fields.extend(field for field in ("v_x", "v_y") if field in getattr(dh, "field_names", ()))
    if "vP" in selected:
        porous_fields.extend(field for field in ("vP_x", "vP_y") if field in getattr(dh, "field_names", ()))
    if "vS" in selected:
        porous_fields.extend(field for field in ("vS_x", "vS_y") if field in getattr(dh, "field_names", ()))
    if "u" in selected:
        porous_fields.extend(field for field in ("u_x", "u_y") if field in getattr(dh, "field_names", ()))
    if "phi" in selected:
        porous_fields.extend(field for field in ("phi",) if field in getattr(dh, "field_names", ()))
    fluid_expected = {field: _field_dofs_by_alpha(field, threshold=alpha_high, mode="high") for field in fluid_fields}
    porous_expected = {field: _field_dofs_by_alpha(field, threshold=alpha_low, mode="low") for field in porous_fields}
    return alpha_low, alpha_high, fluid_expected, porous_expected


def _assemble_small_benchmark7_reduced(
    backend: str,
    *,
    mechanics_nondim_mode: str = "legacy",
    drag_formulation: str = "direct",
    return_problem: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[dict[str, object], np.ndarray, np.ndarray]:
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
        enable_phi_evolution=True,
        drag_formulation=str(drag_formulation),
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.05, phi_b=0.18)

    qdeg = 6
    dt_c = Constant(0.025)
    forms = _build_forms(
        problem,
        qdeg=qdeg,
        dt_c=dt_c,
        theta=1.0,
        rho_f=1.0,
        mu_f=0.035,
        mu_b=0.035,
        mu_b_model="mu",
        kappa_inv=1.0e3,
        mu_s=1.67785e5,
        lambda_s=8.22148e6,
        phi_b=0.18,
        M_alpha=1.0,
        gamma_alpha=1.0,
        eps_alpha=0.05,
        solid_visco_eta=0.0,
        gamma_div=0.0,
        mechanics_nondim_mode=str(mechanics_nondim_mode),
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
        alpha_supg=0.5,
        alpha_cip=0.0,
        alpha_regularization="ch",
        alpha_reg_gamma=1.0,
        alpha_reg_eps_normal=0.05,
        alpha_reg_eps_tangent=0.0125,
        alpha_reg_eta=1.0e-12,
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        support_physics="internal_conversion",
        solid_model="linear",
        kappa_inv_model="refmap",
        drag_formulation=str(drag_formulation),
        fluid_convection="full",
        skeleton_pressure_mode="whole_domain",
        alpha_biot=None,
        enable_phi_evolution=True,
        include_skeleton_acceleration=True,
        rho_s0_tilde=1.1,
        skeleton_inertia_convection="lagged",
        ds_hdiv_tangential=None,
        ds_alpha_transport=None,
        ds_B_transport=None,
        hdiv_tangential_gamma=20.0,
        hdiv_tangential_method="penalty",
    )

    bcs = _build_bcs(
        fluid_space="cg",
        enable_phi_evolution=True,
        y_interface=1.0,
        eps_alpha=0.05,
        v_in=5.0,
        t_ramp=0.02,
        alpha_bc_mode="natural",
        solid_bc_mode="base_only",
    )
    bcs_now = NewtonSolver._freeze_bcs(bcs, 0.025)
    bcs_homog = [
        BoundaryCondition(b.field, b.method, b.domain_tag, (lambda x, y: 0.0))
        for b in bcs_now
    ]

    solver = NewtonSolver(
        forms.residual_form,
        forms.jacobian_form,
        dof_handler=problem["dh"],
        mixed_element=problem["me"],
        bcs=bcs_now,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(
            newton_tol=1.0e-10,
            newton_rtol=1.0e-10,
            max_newton_iter=2,
            line_search=False,
            ls_mode="dealii",
        ),
        lin_params=LinearSolverParameters(backend="scipy", tol=1.0e-10, maxit=5000),
        quad_order=qdeg,
        backend=backend,
    )
    solver._current_bcs = bcs_now
    problem["_reduced_active_dofs"] = np.asarray(solver.active_dofs, dtype=int)

    funcs = [
        problem["v_k"],
        problem["p_k"],
        problem["vS_k"],
        problem["u_k"],
        problem["alpha_k"],
        problem["mu_k"],
        problem["phi_k"],
        problem["S_k"],
    ]
    if problem.get("lambda_drag_k") is not None:
        funcs.insert(4, problem["lambda_drag_k"])
    prev_funcs = [
        problem["v_n"],
        problem["p_n"],
        problem["vS_n"],
        problem["u_n"],
        problem["alpha_n"],
        problem["mu_n"],
        problem["phi_n"],
        problem["S_n"],
    ]
    if problem.get("lambda_drag_n") is not None:
        prev_funcs.insert(4, problem["lambda_drag_n"])
    coeffs = {f.name: f for f in funcs}
    coeffs.update({f.name: f for f in prev_funcs})
    coeffs["dt"] = dt_c

    A_red, R_red = solver._assemble_system_reduced(coeffs, need_matrix=True)
    if bool(return_problem):
        return problem, A_red.toarray(), np.asarray(R_red, dtype=float)
    return A_red.toarray(), np.asarray(R_red, dtype=float)


def _assemble_small_benchmark7_split_reduced(
    backend: str,
    *,
    mechanics_nondim_mode: str,
    drag_formulation: str = "direct",
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
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
        pressure_mean_constraint=False,
        solid_volumetric_split=True,
        drag_formulation=str(drag_formulation),
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.05, phi_b=0.18)

    qdeg = 6
    dt_c = Constant(0.025)
    forms = _build_forms(
        problem,
        qdeg=qdeg,
        dt_c=dt_c,
        theta=1.0,
        rho_f=1.0,
        mu_f=0.035,
        mu_b=0.035,
        mu_b_model="mu",
        kappa_inv=1.0e3,
        mu_s=1.67785e5,
        lambda_s=8.22148e6,
        phi_b=0.18,
        M_alpha=1.0,
        gamma_alpha=1.0,
        eps_alpha=0.05,
        solid_visco_eta=0.0,
        gamma_div=0.0,
        mechanics_nondim_mode=str(mechanics_nondim_mode),
        solid_volumetric_split=True,
        solid_volumetric_penalty=1.0,
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
        alpha_supg=0.5,
        alpha_cip=0.0,
        alpha_regularization="ch",
        alpha_reg_gamma=1.0,
        alpha_reg_eps_normal=0.05,
        alpha_reg_eps_tangent=0.0125,
        alpha_reg_eta=1.0e-12,
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        support_physics="internal_conversion",
        solid_model="linear",
        kappa_inv_model="refmap",
        drag_formulation=str(drag_formulation),
        fluid_convection="full",
        skeleton_pressure_mode="seboldt",
        alpha_biot=1.0,
        enable_phi_evolution=False,
        include_skeleton_acceleration=True,
        rho_s0_tilde=1.1,
        skeleton_inertia_convection="lagged",
    )

    bcs = _build_bcs(
        fluid_space="cg",
        enable_phi_evolution=False,
        y_interface=1.0,
        eps_alpha=0.05,
        v_in=5.0,
        t_ramp=0.02,
        alpha_bc_mode="natural",
        solid_bc_mode="base_only",
        pressure_mean_constraint=False,
    )
    bcs_now = NewtonSolver._freeze_bcs(bcs, 0.025)
    bcs_homog = [
        BoundaryCondition(b.field, b.method, b.domain_tag, (lambda x, y: 0.0))
        for b in bcs_now
    ]

    solver = NewtonSolver(
        forms.residual_form,
        forms.jacobian_form,
        dof_handler=problem["dh"],
        mixed_element=problem["me"],
        bcs=bcs_now,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(
            newton_tol=1.0e-10,
            newton_rtol=1.0e-10,
            max_newton_iter=2,
            line_search=False,
            ls_mode="dealii",
        ),
        lin_params=LinearSolverParameters(backend="scipy", tol=1.0e-10, maxit=5000),
        quad_order=qdeg,
        backend=backend,
    )
    solver._current_bcs = bcs_now
    problem["_reduced_active_dofs"] = np.asarray(solver.active_dofs, dtype=int)

    funcs = [
        problem["v_k"],
        problem["p_k"],
        problem["pi_s_k"],
        problem["vS_k"],
        problem["u_k"],
        problem["alpha_k"],
        problem["mu_k"],
    ]
    if problem.get("lambda_drag_k") is not None:
        funcs.insert(5, problem["lambda_drag_k"])
    prev_funcs = [
        problem["v_n"],
        problem["p_n"],
        problem["pi_s_n"],
        problem["vS_n"],
        problem["u_n"],
        problem["alpha_n"],
        problem["mu_n"],
    ]
    if problem.get("lambda_drag_n") is not None:
        prev_funcs.insert(5, problem["lambda_drag_n"])
    coeffs = {f.name: f for f in funcs}
    coeffs.update({f.name: f for f in prev_funcs})
    coeffs["dt"] = dt_c

    A_red, R_red = solver._assemble_system_reduced(coeffs, need_matrix=True)
    return problem, A_red.toarray(), np.asarray(R_red, dtype=float)


def _reduced_field_indices(problem: dict[str, object], field: str) -> np.ndarray:
    free = np.asarray(problem.get("_reduced_active_dofs", ()), dtype=int)
    if free.size == 0:
        free = np.asarray(problem["dh"].free_dofs, dtype=int)
    g2r = {int(g): i for i, g in enumerate(free.tolist())}
    sl = np.asarray(problem["dh"].get_field_slice(field), dtype=int)
    return np.asarray([g2r[int(g)] for g in sl.tolist() if int(g) in g2r], dtype=int)


@pytest.mark.parametrize("backend", ("cpp",))
def test_benchmark7_solver_reduced_backend_matches_python(monkeypatch, tmp_path, backend):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"jit_cache_{backend}"))
    if backend == "cpp":
        monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")
    else:
        monkeypatch.delenv("PYCUTFEM_JIT_BACKEND", raising=False)

    A_ref, R_ref = _assemble_small_benchmark7_reduced("python")
    A, R = _assemble_small_benchmark7_reduced(backend)

    np.testing.assert_allclose(A, A_ref, rtol=1.0e-9, atol=1.0e-9)
    np.testing.assert_allclose(R, R_ref, rtol=1.0e-9, atol=1.0e-9)


@pytest.mark.parametrize("backend", ("cpp",))
def test_benchmark7_mixed_drag_reduced_backend_matches_python(monkeypatch, tmp_path, backend):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"jit_cache_drag_mixed_{backend}"))
    if backend == "cpp":
        monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")
    else:
        monkeypatch.delenv("PYCUTFEM_JIT_BACKEND", raising=False)

    A_ref, R_ref = _assemble_small_benchmark7_reduced("python", drag_formulation="mixed_lm")
    A, R = _assemble_small_benchmark7_reduced(backend, drag_formulation="mixed_lm")

    np.testing.assert_allclose(A, A_ref, rtol=1.0e-9, atol=1.0e-9)
    np.testing.assert_allclose(R, R_ref, rtol=1.0e-9, atol=1.0e-9)


@pytest.mark.parametrize("backend", ("cpp",))
def test_benchmark7_condition_balanced_split_backend_matches_python(monkeypatch, tmp_path, backend):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"jit_cache_split_{backend}"))
    if backend == "cpp":
        monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")
    else:
        monkeypatch.delenv("PYCUTFEM_JIT_BACKEND", raising=False)

    _problem_ref, A_ref, R_ref = _assemble_small_benchmark7_split_reduced(
        "python",
        mechanics_nondim_mode="condition_balanced",
    )
    _problem, A, R = _assemble_small_benchmark7_split_reduced(
        backend,
        mechanics_nondim_mode="condition_balanced",
    )

    np.testing.assert_allclose(A, A_ref, rtol=1.0e-9, atol=1.0e-9)
    np.testing.assert_allclose(R, R_ref, rtol=1.0e-9, atol=1.0e-9)


@pytest.mark.parametrize("backend", ("cpp",))
def test_benchmark7_mixed_drag_condition_balanced_split_backend_matches_python(monkeypatch, tmp_path, backend):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"jit_cache_drag_mixed_split_{backend}"))
    if backend == "cpp":
        monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")
    else:
        monkeypatch.delenv("PYCUTFEM_JIT_BACKEND", raising=False)

    _problem_ref, A_ref, R_ref = _assemble_small_benchmark7_split_reduced(
        "python",
        mechanics_nondim_mode="condition_balanced",
        drag_formulation="mixed_lm",
    )
    _problem, A, R = _assemble_small_benchmark7_split_reduced(
        backend,
        mechanics_nondim_mode="condition_balanced",
        drag_formulation="mixed_lm",
    )

    np.testing.assert_allclose(A, A_ref, rtol=1.0e-9, atol=1.0e-9)
    np.testing.assert_allclose(R, R_ref, rtol=1.0e-9, atol=1.0e-9)


def test_benchmark7_condition_balanced_coordinate_scaling_reduces_small_operator_cond():
    problem, A_raw, _ = _assemble_small_benchmark7_reduced(
        "python",
        mechanics_nondim_mode="stress_balance",
        return_problem=True,
    )
    scales = _condition_balanced_field_scales(
        mechanics_nondim_mode="condition_balanced",
        dt=0.025,
        mu_f=0.035,
        kappa_inv=1.0e3,
        mu_s=1.67785e5,
        lambda_s=8.22148e6,
        rho_s0_tilde=1.1,
        dim=2,
    )
    d = _reduced_field_scale_vector(problem, scales)
    A_scaled = d[:, None] * A_raw * d[None, :]

    assert np.linalg.cond(A_scaled) < np.linalg.cond(A_raw)


def test_benchmark7_mixed_drag_ruiz_scaling_reduces_small_operator_cond():
    A_raw, _ = _assemble_small_benchmark7_reduced(
        "python",
        mechanics_nondim_mode="condition_balanced",
        drag_formulation="mixed_lm",
    )
    A_csr = sp.csr_matrix(A_raw)
    row_scale, col_scale = NewtonSolver._direct_solve_ruiz_scaling(A_csr, iters=8)
    A_scaled = (sp.diags(row_scale, format="csr") @ A_csr @ sp.diags(col_scale, format="csr")).toarray()

    assert np.linalg.cond(A_scaled) < 0.05 * np.linalg.cond(A_raw)


def test_benchmark7_condition_balanced_auto_kinematic_stack_keeps_constant_row_scale():
    lo = _condition_balanced_kinematic_setup(
        mechanics_nondim_mode="condition_balanced",
        mu_f=0.035,
        kappa_inv=1.0e3,
        gamma_u=0.0,
        u_extension_mode="l2",
        gamma_u_pin=0.0,
        gamma_vS=None,
        vS_extension_mode=None,
        gamma_vS_pin=None,
    )
    hi = _condition_balanced_kinematic_setup(
        mechanics_nondim_mode="condition_balanced",
        mu_f=0.035,
        kappa_inv=1.0e5,
        gamma_u=0.0,
        u_extension_mode="l2",
        gamma_u_pin=0.0,
        gamma_vS=None,
        vS_extension_mode=None,
        gamma_vS_pin=None,
    )

    assert lo["gamma_u"] == pytest.approx(1.0)
    assert lo["u_extension_mode"] == "grad"
    assert lo["gamma_u_pin"] == pytest.approx(1.0e-6)
    assert lo["gamma_vS"] == pytest.approx(1.0)
    assert lo["vS_extension_mode"] == "grad"
    assert lo["gamma_vS_pin"] == pytest.approx(1.0e-6)
    assert lo["kinematics_scale"] == pytest.approx(1.0)
    assert hi["kinematics_scale"] == pytest.approx(1.0)


def test_benchmark7_condition_balanced_auto_volume_al_tracks_kappa():
    lo = _condition_balanced_volume_setup(
        mechanics_nondim_mode="condition_balanced",
        mu_f=0.035,
        kappa_inv=1.0e3,
        gamma_div=0.0,
        auto_gamma_div=True,
    )
    hi = _condition_balanced_volume_setup(
        mechanics_nondim_mode="condition_balanced",
        mu_f=0.035,
        kappa_inv=1.0e5,
        gamma_div=0.0,
        auto_gamma_div=True,
    )

    assert lo["gamma_div"] == pytest.approx(1.0 / 35.0)
    assert hi["gamma_div"] == pytest.approx(1.0 / 3500.0)


def test_benchmark7_condition_balanced_solid_cut_fix_uses_hard_y_cut():
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=4,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=False,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.05, phi_b=0.18)
    y_cut = _condition_balanced_solid_cutoff_y(
        mechanics_nondim_mode="condition_balanced",
        y_interface=1.0,
        solid_dof_y_cut=None,
        condition_balanced_solid_cut_fix=True,
    )
    counts = _tag_inactive_solid_dofs_above_y(problem, y_cut=y_cut)

    assert y_cut == pytest.approx(1.0)
    inactive = set(int(d) for d in problem["dh"].dof_tags.get("inactive", set()))
    expected_flat: set[int] = set()
    for field in ("u_x", "u_y", "vS_x", "vS_y"):
        sl = np.asarray(problem["dh"].get_field_slice(field), dtype=int)
        xy = np.asarray(problem["dh"].get_dof_coords(field), dtype=float)
        field_expected = set(int(g) for g in sl[xy[:, 1] > 1.0 + 1.0e-12].tolist())
        assert counts[field] == len(field_expected)
        expected_flat.update(field_expected)
    assert str(problem["_inactive_solid_alpha_phase"]) == "static_y_cut"
    assert problem["_inactive_solid_reference_y"] is None
    assert inactive == expected_flat
    assert inactive
    solid_above: set[int] = set()
    for field in ("u_x", "u_y", "vS_x", "vS_y"):
        sl = np.asarray(problem["dh"].get_field_slice(field), dtype=int)
        xy = np.asarray(problem["dh"].get_dof_coords(field), dtype=float)
        solid_above.update(int(g) for g in sl[xy[:, 1] > 1.0 + 1.0e-12].tolist())
    assert solid_above
    assert inactive == solid_above
    assert not (inactive & set(np.asarray(problem["dh"].get_field_slice("p"), dtype=int).tolist()))
    assert not (inactive & set(np.asarray(problem["dh"].get_field_slice("alpha"), dtype=int).tolist()))


def test_benchmark7_condition_balanced_solid_interface_band_retags_and_reactivates_new_dofs():
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=4,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=False,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.05, phi_b=0.18)
    y_cut = _condition_balanced_solid_cutoff_y(
        mechanics_nondim_mode="condition_balanced",
        y_interface=1.0,
        solid_dof_y_cut=None,
        condition_balanced_solid_cut_fix=True,
    )
    problem["_inactive_solid_reference_y"] = float(y_cut)
    problem["_inactive_solid_alpha_band_halfwidth"] = 0.25
    _tag_inactive_solid_dofs_outside_interface_band(problem, reference_y=y_cut, alpha_state_key="alpha_k")
    inactive_initial = set(int(d) for d in problem["dh"].dof_tags.get("inactive", set()))

    problem["alpha_k"].set_values_from_function(
        lambda x, y: float(_alpha_equilibrium(y, y_interface=1.25, eps_alpha=0.05))
    )
    counts = _tag_inactive_solid_dofs_outside_interface_band(
        problem,
        reference_y=y_cut,
        alpha_state_key="alpha_k",
    )
    phase, expected = _expected_interface_band_inactive_solid_dofs(problem, reference_y=y_cut, band_halfwidth=0.25)
    inactive_new = set(int(d) for d in problem["dh"].dof_tags.get("inactive", set()))

    assert str(problem["_inactive_solid_alpha_phase"]) == phase
    assert inactive_initial
    assert inactive_new != inactive_initial
    assert inactive_initial - inactive_new
    expected_flat: set[int] = set()
    for field in ("u_x", "u_y", "vS_x", "vS_y"):
        field_expected = set(expected[field])
        assert counts[field] == len(field_expected)
        expected_flat.update(field_expected)
    assert inactive_new == expected_flat


def test_benchmark7_condition_balanced_solver_refresh_retargets_active_dofs_when_interface_moves():
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=4,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=False,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.05, phi_b=0.18)
    qdeg = 6
    dt_c = Constant(0.025)
    forms = _build_forms(
        problem,
        qdeg=qdeg,
        dt_c=dt_c,
        theta=1.0,
        rho_f=1.0,
        mu_f=0.035,
        mu_b=0.035,
        mu_b_model="mu",
        kappa_inv=1.0e3,
        mu_s=1.67785e5,
        lambda_s=8.22148e6,
        phi_b=0.18,
        M_alpha=1.0,
        gamma_alpha=1.0,
        eps_alpha=0.05,
        solid_visco_eta=0.0,
        gamma_div=0.0,
        mechanics_nondim_mode="condition_balanced",
        gamma_v=0.0,
        v_extension_mode="l2",
        gamma_v_pin=0.0,
        gamma_p=0.0,
        p_extension_mode="l2",
        gamma_p_pin=0.0,
        gamma_vP=0.0,
        vP_extension_mode="l2",
        gamma_vP_pin=0.0,
        gamma_p_pore=0.0,
        p_pore_extension_mode="l2",
        gamma_p_pore_pin=0.0,
        gamma_rho_s=0.0,
        rho_s_extension_mode="l2",
        gamma_rho_s_pin=0.0,
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
        alpha_supg=0.5,
        alpha_cip=0.0,
        alpha_regularization="ch",
        alpha_reg_gamma=1.0,
        alpha_reg_eps_normal=0.05,
        alpha_reg_eps_tangent=0.0125,
        alpha_reg_eta=1.0e-12,
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        support_physics="internal_conversion",
        solid_model="linear",
        kappa_inv_model="refmap",
        fluid_convection="lagged",
        skeleton_pressure_mode="whole_domain",
        alpha_biot=None,
        enable_phi_evolution=False,
        include_skeleton_acceleration=True,
        rho_s0_tilde=1.1,
        skeleton_inertia_convection="lagged",
        ds_hdiv_tangential=None,
        ds_alpha_transport=None,
        ds_B_transport=None,
        hdiv_tangential_gamma=20.0,
        hdiv_tangential_method="penalty",
    )
    bcs = _build_bcs(
        fluid_space="cg",
        enable_phi_evolution=False,
        y_interface=1.0,
        eps_alpha=0.05,
        v_in=5.0,
        t_ramp=0.02,
        alpha_bc_mode="natural",
        solid_bc_mode="base_only",
    )
    bcs_now = NewtonSolver._freeze_bcs(bcs, 0.025)
    bcs_homog = [
        BoundaryCondition(b.field, b.method, b.domain_tag, (lambda x, y: 0.0))
        for b in bcs_now
    ]
    solver = NewtonSolver(
        forms.residual_form,
        forms.jacobian_form,
        dof_handler=problem["dh"],
        mixed_element=problem["me"],
        bcs=bcs_now,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(
            newton_tol=1.0e-10,
            newton_rtol=1.0e-10,
            max_newton_iter=2,
            line_search=False,
            ls_mode="dealii",
        ),
        lin_params=LinearSolverParameters(backend="scipy", tol=1.0e-10, maxit=5000),
        quad_order=qdeg,
        backend="python",
    )
    _set_solver_active_fields_with_tracking(solver, ("u_x", "u_y", "vS_x", "vS_y"))
    y_cut = _condition_balanced_solid_cutoff_y(
        mechanics_nondim_mode="condition_balanced",
        y_interface=1.0,
        solid_dof_y_cut=None,
        condition_balanced_solid_cut_fix=True,
    )
    problem["_inactive_solid_reference_y"] = float(y_cut)
    problem["_inactive_solid_alpha_band_halfwidth"] = 0.25
    _refresh_solver_inactive_solid_interface_band(
        problem=problem,
        target_solver=solver,
        reference_y=y_cut,
        alpha_state_key="alpha_k",
    )
    n_active_initial = int(solver.active_dofs.size)

    problem["alpha_k"].set_values_from_function(
        lambda x, y: float(_alpha_equilibrium(y, y_interface=1.25, eps_alpha=0.05))
    )
    _refresh_solver_inactive_solid_interface_band(
        problem=problem,
        target_solver=solver,
        reference_y=y_cut,
        alpha_state_key="alpha_k",
    )

    assert int(solver.active_dofs.size) > n_active_initial


def test_benchmark7_final_form_internal_zero_dirichlet_tags_offdomain_dofs() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=4,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
        one_domain_formulation="final_form",
        full_ratio_free_state=True,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.05, phi_b=0.18)
    problem["final_form_internal_zero_dirichlet"] = True
    problem["final_form_internal_zero_dirichlet_distance_multiple"] = 1.5

    info = _tag_final_form_internal_zero_dirichlet_dofs(problem, distance_multiple=1.5)
    alpha_low, alpha_high, fluid_expected, porous_expected = _expected_final_form_internal_zero_dofs(
        problem,
        distance_multiple=1.5,
    )

    assert info["alpha_low"] == pytest.approx(alpha_low)
    assert info["alpha_high"] == pytest.approx(alpha_high)

    fluid_tagged = set(int(d) for d in problem["dh"].dof_tags.get(_FINAL_FORM_INTERNAL_ZERO_FLUID_TAG, set()))
    porous_tagged = set(int(d) for d in problem["dh"].dof_tags.get(_FINAL_FORM_INTERNAL_ZERO_POROUS_TAG, set()))
    fluid_expected_flat: set[int] = set()
    porous_expected_flat: set[int] = set()
    for field, expected in fluid_expected.items():
        assert int(info["fluid_counts"][field]) == len(expected)
        fluid_expected_flat.update(int(d) for d in expected)
    for field, expected in porous_expected.items():
        assert int(info["porous_counts"][field]) == len(expected)
        porous_expected_flat.update(int(d) for d in expected)
    assert fluid_tagged == fluid_expected_flat
    assert porous_tagged == porous_expected_flat
    assert fluid_tagged
    assert porous_tagged


def test_benchmark7_final_form_internal_zero_dirichlet_tags_can_limit_to_vf_only() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=4,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
        one_domain_formulation="final_form",
        full_ratio_free_state=True,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.05, phi_b=0.18)
    problem["final_form_internal_zero_dirichlet"] = True
    problem["final_form_internal_zero_dirichlet_distance_multiple"] = 1.5
    problem["final_form_internal_zero_dirichlet_targets"] = ("vf",)

    info = _tag_final_form_internal_zero_dirichlet_dofs(problem, distance_multiple=1.5)
    _, _, fluid_expected, porous_expected = _expected_final_form_internal_zero_dofs(
        problem,
        distance_multiple=1.5,
        targets=("vf",),
    )

    fluid_tagged = set(int(d) for d in problem["dh"].dof_tags.get(_FINAL_FORM_INTERNAL_ZERO_FLUID_TAG, set()))
    porous_tagged = set(int(d) for d in problem["dh"].dof_tags.get(_FINAL_FORM_INTERNAL_ZERO_POROUS_TAG, set()))
    fluid_expected_flat: set[int] = set()
    for field, expected in fluid_expected.items():
        assert int(info["fluid_counts"][field]) == len(expected)
        fluid_expected_flat.update(int(d) for d in expected)
    assert info["porous_counts"] == {}
    assert fluid_tagged == fluid_expected_flat
    assert porous_tagged == set()
    assert porous_expected == {}
    assert list(info["targets"]) == ["vf"]


def test_benchmark7_final_form_internal_zero_dirichlet_zeroes_tagged_fluid_state_values() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=4,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
        one_domain_formulation="final_form",
        full_ratio_free_state=True,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.05, phi_b=0.18)
    problem["final_form_internal_zero_dirichlet"] = True
    problem["final_form_internal_zero_dirichlet_distance_multiple"] = 1.5
    problem["final_form_internal_zero_dirichlet_targets"] = ("vf",)

    _tag_final_form_internal_zero_dirichlet_dofs(problem, distance_multiple=1.5)
    fluid_tagged = np.asarray(
        sorted(int(d) for d in problem["dh"].dof_tags.get(_FINAL_FORM_INTERNAL_ZERO_FLUID_TAG, set())),
        dtype=int,
    )
    assert fluid_tagged.size > 0
    assert np.linalg.norm(problem["v_k"].get_nodal_values(fluid_tagged), ord=np.inf) > 0.0
    assert np.linalg.norm(problem["v_n"].get_nodal_values(fluid_tagged), ord=np.inf) > 0.0

    _zero_tagged_internal_zero_dirichlet_state(problem)

    np.testing.assert_allclose(problem["v_k"].get_nodal_values(fluid_tagged), 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(problem["v_n"].get_nodal_values(fluid_tagged), 0.0, atol=0.0, rtol=0.0)


def test_benchmark7_final_form_internal_zero_dirichlet_tags_phi_on_free_side() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=4,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
        one_domain_formulation="final_form",
        full_ratio_free_state=True,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.05, phi_b=0.18)
    problem["final_form_internal_zero_dirichlet"] = True
    problem["final_form_internal_zero_dirichlet_distance_multiple"] = 1.5
    problem["final_form_internal_zero_dirichlet_targets"] = ("phi",)

    info = _tag_final_form_internal_zero_dirichlet_dofs(problem, distance_multiple=1.5)
    _, _, fluid_expected, porous_expected = _expected_final_form_internal_zero_dofs(
        problem,
        distance_multiple=1.5,
        targets=("phi",),
    )

    assert info["fluid_counts"] == {}
    assert int(info["porous_counts"]["phi"]) == len(porous_expected["phi"])
    fluid_tagged = set(int(d) for d in problem["dh"].dof_tags.get(_FINAL_FORM_INTERNAL_ZERO_FLUID_TAG, set()))
    porous_tagged = set(int(d) for d in problem["dh"].dof_tags.get(_FINAL_FORM_INTERNAL_ZERO_POROUS_TAG, set()))
    assert fluid_tagged == set()
    assert porous_tagged == set(int(d) for d in porous_expected["phi"])
    assert fluid_expected == {}
    assert list(info["targets"]) == ["phi"]


def test_benchmark7_final_form_internal_zero_dirichlet_sets_phi_to_one_on_tagged_state_values() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=4,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
        one_domain_formulation="final_form",
        full_ratio_free_state=True,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.05, phi_b=0.18)
    problem["final_form_internal_zero_dirichlet"] = True
    problem["final_form_internal_zero_dirichlet_distance_multiple"] = 1.5
    problem["final_form_internal_zero_dirichlet_targets"] = ("phi",)

    _tag_final_form_internal_zero_dirichlet_dofs(problem, distance_multiple=1.5)
    porous_tagged = np.asarray(
        sorted(int(d) for d in problem["dh"].dof_tags.get(_FINAL_FORM_INTERNAL_ZERO_POROUS_TAG, set())),
        dtype=int,
    )
    assert porous_tagged.size > 0
    assert np.linalg.norm(problem["phi_k"].get_nodal_values(porous_tagged) - 1.0, ord=np.inf) > 0.0
    assert np.linalg.norm(problem["phi_n"].get_nodal_values(porous_tagged) - 1.0, ord=np.inf) > 0.0

    _zero_tagged_internal_zero_dirichlet_state(problem)

    np.testing.assert_allclose(problem["phi_k"].get_nodal_values(porous_tagged), 1.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(problem["phi_n"].get_nodal_values(porous_tagged), 1.0, atol=0.0, rtol=0.0)


def test_benchmark7_final_form_internal_zero_dirichlet_build_bcs_supports_phi_target() -> None:
    bcs = _build_bcs(
        fluid_space="cg",
        enable_phi_evolution=True,
        one_domain_formulation="final_form",
        full_ratio_free_state=True,
        y_interface=1.0,
        eps_alpha=0.05,
        v_in=5.0,
        t_ramp=0.02,
        alpha_bc_mode="natural",
        solid_bc_mode="base_only",
        final_form_internal_zero_dirichlet=True,
        final_form_internal_zero_targets=("phi",),
    )
    phi_tagged = [
        bc for bc in bcs
        if bc.field == "phi" and bc.method == "dirichlet" and bc.domain_tag == _FINAL_FORM_INTERNAL_ZERO_POROUS_TAG
    ]
    assert len(phi_tagged) == 1
    assert float(phi_tagged[0].value(0.3, 1.2, 0.0)) == pytest.approx(1.0)


def test_benchmark7_final_form_internal_zero_dirichlet_refresh_updates_active_dofs() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=4,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
        one_domain_formulation="final_form",
        full_ratio_free_state=True,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.05, phi_b=0.18)
    problem["final_form_internal_zero_dirichlet"] = True
    problem["final_form_internal_zero_dirichlet_distance_multiple"] = 1.5
    _tag_final_form_internal_zero_dirichlet_dofs(problem, distance_multiple=1.5)

    qdeg = 6
    dt_c = Constant(0.025)
    forms = _build_forms(
        problem,
        qdeg=qdeg,
        dt_c=dt_c,
        theta=1.0,
        rho_f=1.0,
        mu_f=0.035,
        mu_b=0.035,
        mu_b_model="mu",
        kappa_inv=1.0e3,
        mu_s=1.67785e5,
        lambda_s=8.22148e6,
        phi_b=0.18,
        M_alpha=1.0,
        gamma_alpha=1.0,
        eps_alpha=0.05,
        solid_visco_eta=0.0,
        gamma_div=0.0,
        mechanics_nondim_mode="legacy",
        gamma_v=0.0,
        v_extension_mode="l2",
        gamma_v_pin=0.0,
        gamma_p=0.0,
        p_extension_mode="l2",
        gamma_p_pin=0.0,
        gamma_vP=0.0,
        vP_extension_mode="l2",
        gamma_vP_pin=0.0,
        gamma_p_pore=0.0,
        p_pore_extension_mode="l2",
        gamma_p_pore_pin=0.0,
        gamma_rho_s=0.0,
        rho_s_extension_mode="l2",
        gamma_rho_s_pin=0.0,
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
        alpha_supg=0.5,
        alpha_cip=0.0,
        alpha_regularization="ch",
        alpha_reg_gamma=1.0,
        alpha_reg_eps_normal=0.05,
        alpha_reg_eps_tangent=0.0125,
        alpha_reg_eta=1.0e-12,
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        support_physics="stored_support",
        solid_model="linear",
        kappa_inv_model="refmap",
        reduced_support_state="alpha_B",
        stored_support_content_mode="evolve_B",
        drag_formulation="direct",
        full_ratio_free_state=True,
        fluid_convection="lagged",
        skeleton_pressure_mode="whole_domain",
        alpha_biot=None,
        enable_phi_evolution=True,
        include_skeleton_acceleration=True,
        rho_s0_tilde=1.1,
        skeleton_inertia_convection="lagged",
        final_form_constant_rho_s=True,
        ds_hdiv_tangential=None,
        ds_alpha_transport=None,
        ds_B_transport=None,
        hdiv_tangential_gamma=20.0,
        hdiv_tangential_method="penalty",
    )
    bcs = _build_bcs(
        fluid_space="cg",
        enable_phi_evolution=True,
        one_domain_formulation="final_form",
        full_ratio_free_state=True,
        y_interface=1.0,
        eps_alpha=0.05,
        v_in=5.0,
        t_ramp=0.02,
        alpha_bc_mode="natural",
        solid_bc_mode="base_only",
        final_form_internal_zero_dirichlet=True,
    )
    bcs_now = NewtonSolver._freeze_bcs(bcs, 0.025)
    bcs_homog = [
        BoundaryCondition(b.field, b.method, b.domain_tag, (lambda x, y: 0.0))
        for b in bcs_now
    ]
    solver = NewtonSolver(
        forms.residual_form,
        forms.jacobian_form,
        dof_handler=problem["dh"],
        mixed_element=problem["me"],
        bcs=bcs_now,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(
            newton_tol=1.0e-10,
            newton_rtol=1.0e-10,
            max_newton_iter=2,
            line_search=False,
            ls_mode="dealii",
        ),
        lin_params=LinearSolverParameters(backend="scipy", tol=1.0e-10, maxit=5000),
        quad_order=qdeg,
        backend="python",
    )
    tracked_fields = ("v_x", "v_y", "vP_x", "vP_y", "vS_x", "vS_y", "u_x", "u_y")
    _set_solver_active_fields_with_tracking(solver, tracked_fields)
    expected0_full = _solver_requested_active_dofs(solver, tracked_fields)
    expected0 = solver._filter_candidate_active_dofs(expected0_full)
    assert np.array_equal(np.asarray(solver.active_dofs, dtype=int), np.asarray(expected0, dtype=int))
    fluid_initial = set(int(d) for d in problem["dh"].dof_tags.get(_FINAL_FORM_INTERNAL_ZERO_FLUID_TAG, set()))
    porous_initial = set(int(d) for d in problem["dh"].dof_tags.get(_FINAL_FORM_INTERNAL_ZERO_POROUS_TAG, set()))

    problem["alpha_k"].set_values_from_function(
        lambda x, y: float(_alpha_equilibrium(y, y_interface=1.25, eps_alpha=0.05))
    )
    _refresh_solver_final_form_internal_zero_dirichlet(
        problem=problem,
        target_solver=solver,
        alpha_state_key="alpha_k",
    )

    fluid_new = set(int(d) for d in problem["dh"].dof_tags.get(_FINAL_FORM_INTERNAL_ZERO_FLUID_TAG, set()))
    porous_new = set(int(d) for d in problem["dh"].dof_tags.get(_FINAL_FORM_INTERNAL_ZERO_POROUS_TAG, set()))
    expected1_full = _solver_requested_active_dofs(solver, tracked_fields)
    expected1 = solver._filter_candidate_active_dofs(expected1_full)
    assert fluid_new != fluid_initial or porous_new != porous_initial
    assert np.array_equal(np.asarray(solver.active_dofs, dtype=int), np.asarray(expected1, dtype=int))


def test_benchmark7_final_form_internal_zero_dirichlet_per_step_binding_keeps_pre_cb_until_refresh() -> None:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.5,
        nx=1,
        ny=4,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
        one_domain_formulation="final_form",
        full_ratio_free_state=True,
    )
    _initialize_small_benchmark7_state(problem, eps_alpha=0.05, phi_b=0.18)
    problem["final_form_internal_zero_dirichlet"] = True
    problem["final_form_internal_zero_dirichlet_distance_multiple"] = 1.5
    problem["final_form_internal_zero_dirichlet_targets"] = ("vf",)
    problem["final_form_internal_zero_dirichlet_retag_mode"] = "per_step"
    _tag_final_form_internal_zero_dirichlet_dofs(problem, distance_multiple=1.5)

    qdeg = 6
    dt_c = Constant(0.025)
    forms = _build_forms(
        problem,
        qdeg=qdeg,
        dt_c=dt_c,
        theta=1.0,
        rho_f=1.0,
        mu_f=0.035,
        mu_b=0.035,
        mu_b_model="mu",
        kappa_inv=1.0e3,
        mu_s=1.67785e5,
        lambda_s=8.22148e6,
        phi_b=0.18,
        M_alpha=1.0,
        gamma_alpha=1.0,
        eps_alpha=0.05,
        solid_visco_eta=0.0,
        gamma_div=0.0,
        mechanics_nondim_mode="legacy",
        gamma_v=0.0,
        v_extension_mode="l2",
        gamma_v_pin=0.0,
        gamma_p=0.0,
        p_extension_mode="l2",
        gamma_p_pin=0.0,
        gamma_vP=0.0,
        vP_extension_mode="l2",
        gamma_vP_pin=0.0,
        gamma_p_pore=0.0,
        p_pore_extension_mode="l2",
        gamma_p_pore_pin=0.0,
        gamma_rho_s=0.0,
        rho_s_extension_mode="l2",
        gamma_rho_s_pin=0.0,
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
        alpha_supg=0.5,
        alpha_cip=0.0,
        alpha_regularization="ch",
        alpha_reg_gamma=1.0,
        alpha_reg_eps_normal=0.05,
        alpha_reg_eps_tangent=0.0125,
        alpha_reg_eta=1.0e-12,
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        support_physics="stored_support",
        solid_model="linear",
        kappa_inv_model="refmap",
        reduced_support_state="alpha_B",
        stored_support_content_mode="evolve_B",
        drag_formulation="direct",
        full_ratio_free_state=True,
        fluid_convection="lagged",
        skeleton_pressure_mode="whole_domain",
        alpha_biot=None,
        enable_phi_evolution=True,
        include_skeleton_acceleration=True,
        rho_s0_tilde=1.1,
        skeleton_inertia_convection="lagged",
        final_form_constant_rho_s=True,
        ds_hdiv_tangential=None,
        ds_alpha_transport=None,
        ds_B_transport=None,
        hdiv_tangential_gamma=20.0,
        hdiv_tangential_method="penalty",
    )
    bcs = _build_bcs(
        fluid_space="cg",
        enable_phi_evolution=True,
        one_domain_formulation="final_form",
        full_ratio_free_state=True,
        y_interface=1.0,
        eps_alpha=0.05,
        v_in=5.0,
        t_ramp=0.02,
        alpha_bc_mode="natural",
        solid_bc_mode="base_only",
        final_form_internal_zero_dirichlet=True,
        final_form_internal_zero_targets=("vf",),
    )
    bcs_now = NewtonSolver._freeze_bcs(bcs, 0.025)
    bcs_homog = [
        BoundaryCondition(b.field, b.method, b.domain_tag, (lambda x, y: 0.0))
        for b in bcs_now
    ]
    solver = NewtonSolver(
        forms.residual_form,
        forms.jacobian_form,
        dof_handler=problem["dh"],
        mixed_element=problem["me"],
        bcs=bcs_now,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(
            newton_tol=1.0e-10,
            newton_rtol=1.0e-10,
            max_newton_iter=2,
            line_search=False,
            ls_mode="dealii",
        ),
        lin_params=LinearSolverParameters(backend="scipy", tol=1.0e-10, maxit=5000),
        quad_order=qdeg,
        backend="python",
    )
    tracked_fields = ("v_x", "v_y", "vP_x", "vP_y", "vS_x", "vS_y", "u_x", "u_y")
    _set_solver_active_fields_with_tracking(solver, tracked_fields)
    calls: list[int] = []

    def _sentinel_pre_cb(_funcs) -> None:
        calls.append(1)

    solver.pre_cb = _sentinel_pre_cb
    active_initial = np.asarray(solver.active_dofs, dtype=int).copy()
    fluid_initial = set(int(d) for d in problem["dh"].dof_tags.get(_FINAL_FORM_INTERNAL_ZERO_FLUID_TAG, set()))
    _bind_solver_final_form_internal_zero_dirichlet(problem=problem, target_solver=solver)

    assert solver.pre_cb is _sentinel_pre_cb
    problem["alpha_k"].set_values_from_function(
        lambda x, y: float(_alpha_equilibrium(y, y_interface=1.25, eps_alpha=0.05))
    )
    solver.pre_cb(None)
    fluid_after_pre = set(int(d) for d in problem["dh"].dof_tags.get(_FINAL_FORM_INTERNAL_ZERO_FLUID_TAG, set()))
    assert calls == [1]
    assert fluid_after_pre == fluid_initial
    assert np.array_equal(np.asarray(solver.active_dofs, dtype=int), active_initial)

    _refresh_solver_final_form_internal_zero_dirichlet(
        problem=problem,
        target_solver=solver,
        alpha_state_key="alpha_k",
    )

    fluid_after_refresh = set(int(d) for d in problem["dh"].dof_tags.get(_FINAL_FORM_INTERNAL_ZERO_FLUID_TAG, set()))
    assert fluid_after_refresh != fluid_initial
    assert not np.array_equal(np.asarray(solver.active_dofs, dtype=int), active_initial)
