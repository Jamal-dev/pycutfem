import numpy as np

from examples.biofilms.benchmarks.seboldt.paper1_benchmark7_seboldt import _build_forms, _create_problem
from pycutfem.ufl.expressions import Constant
from pycutfem.ufl.forms import Equation, assemble_form


def _phi_profile(x, y):
    return 0.55 + 0.10 * np.asarray(x, dtype=float) - 0.05 * np.asarray(y, dtype=float)


def _vS_profile(x, y):
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    return np.asarray([0.10 * xx, 0.20 * yy], dtype=float)


def _mix_zero_fluid_velocity(x, y):
    phi = _phi_profile(x, y)
    vS = _vS_profile(x, y)
    scale = -(1.0 - phi) / phi
    return np.asarray([scale * vS[0], scale * vS[1]], dtype=float)


def _build_alpha_residual(
    *,
    alpha_advect_with: str,
    alpha_advection_form: str,
    alpha_cip: float = 0.0,
    backend: str = "python",
) -> tuple[object, np.ndarray]:
    problem = _create_problem(
        Lx=1.0,
        Ly=1.0,
        nx=1,
        ny=1,
        poly_order=2,
        pressure_order=1,
        scalar_order=1,
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=True,
    )

    for vf in (problem["v_k"], problem["v_n"]):
        vf.set_values_from_function(_mix_zero_fluid_velocity)
    for vf in (problem["vS_k"], problem["vS_n"]):
        vf.set_values_from_function(_vS_profile)
    for vf in (problem["u_k"], problem["u_n"]):
        vf.nodal_values[:] = 0.0
    for sf in (problem["p_k"], problem["p_n"], problem["mu_k"], problem["mu_n"]):
        sf.nodal_values[:] = 0.0
    for sf in (problem["phi_k"], problem["phi_n"]):
        sf.set_values_from_function(_phi_profile)
    for sf in (problem["alpha_k"], problem["alpha_n"]):
        sf.nodal_values[:] = 1.0
    for sf in (problem["S_k"], problem["S_n"]):
        sf.nodal_values[:] = 1.0

    forms = _build_forms(
        problem,
        qdeg=6,
        dt_c=Constant(0.1),
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
        gamma_u=1.0,
        u_extension_mode="grad",
        gamma_u_pin=1.0e-6,
        u_cip=0.0,
        u_cip_weight="fluid",
        vS_cip=0.0,
        gamma_vS=1.0,
        vS_extension_mode="grad",
        gamma_vS_pin=1.0e-6,
        D_phi=0.0,
        phi_diffusion_weight="fluid",
        gamma_phi=5.0,
        phi_supg=0.0,
        phi_cip=0.0,
        alpha_supg=0.0,
        alpha_cip=float(alpha_cip),
        alpha_regularization="none",
        alpha_reg_gamma=1.0,
        alpha_reg_eps_normal=0.05,
        alpha_reg_eps_tangent=0.0125,
        alpha_reg_eta=1.0e-12,
        alpha_advect_with=str(alpha_advect_with),
        alpha_advection_form=str(alpha_advection_form),
        solid_model="linear",
        kappa_inv_model="refmap",
        enable_phi_evolution=True,
        include_skeleton_acceleration=True,
        rho_s0_tilde=1.1,
        skeleton_inertia_convection="lagged",
    )

    _, residual = assemble_form(
        Equation(None, forms.r_alpha),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=6,
        backend=str(backend),
    )
    return problem["dh"], np.asarray(residual, dtype=float)


def test_one_domain_mix_transport_preserves_uniform_alpha_with_varying_phi() -> None:
    dh_mix, residual_mix = _build_alpha_residual(alpha_advect_with="mix", alpha_advection_form="conservative_weak")
    dh_vs, residual_vs = _build_alpha_residual(alpha_advect_with="vS", alpha_advection_form="conservative")

    alpha_slice_mix = np.asarray(dh_mix.get_field_slice("alpha"), dtype=int)
    alpha_slice_vs = np.asarray(dh_vs.get_field_slice("alpha"), dtype=int)

    mix_norm = np.linalg.norm(residual_mix[alpha_slice_mix], ord=np.inf)
    vs_norm = np.linalg.norm(residual_vs[alpha_slice_vs], ord=np.inf)

    assert mix_norm < 1.0e-5
    assert vs_norm > 1.0e-3
    assert vs_norm / max(mix_norm, 1.0e-16) > 1.0e3


def test_one_domain_alpha_cip_cpp_matches_python_for_scalar_transport() -> None:
    dh_py, residual_py = _build_alpha_residual(
        alpha_advect_with="mix",
        alpha_advection_form="conservative_weak",
        alpha_cip=0.5,
        backend="python",
    )
    dh_cpp, residual_cpp = _build_alpha_residual(
        alpha_advect_with="mix",
        alpha_advection_form="conservative_weak",
        alpha_cip=0.5,
        backend="cpp",
    )

    alpha_slice_py = np.asarray(dh_py.get_field_slice("alpha"), dtype=int)
    alpha_slice_cpp = np.asarray(dh_cpp.get_field_slice("alpha"), dtype=int)

    np.testing.assert_allclose(
        residual_cpp[alpha_slice_cpp],
        residual_py[alpha_slice_py],
        rtol=1.0e-9,
        atol=1.0e-9,
    )
