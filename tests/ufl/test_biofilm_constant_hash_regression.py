from pycutfem.jit.cache import KernelCache
from pycutfem.jit.ir import strip_side_metadata
from pycutfem.jit.visitor import IRGenerator

from tests.test_biofilm_one_domain_jacobian_fd import _build_problem


def _form_hashes(form, dh, *, rank: int) -> tuple[str, ...]:
    irg = IRGenerator()
    cache_sig = (dh.mixed_element.signature(), False, int(rank))
    hashes: list[str] = []
    for integral in form.integrals:
        ir = strip_side_metadata(irg.generate(integral.integrand), on_facet=False)
        hashes.append(KernelCache._hash_ir(ir, cache_sig))
    return tuple(hashes)


def test_biofilm_constant_value_changes_do_not_change_kernel_hashes():
    base_kwargs = dict(
        nx=2,
        ny=2,
        q=4,
        dt_val=0.1,
        rho_f=1.0,
        mu_f=1.0e-2,
        kappa_inv=10.0,
        mu_s=1.0,
        lambda_s=1.0,
        gamma_div=0.1,
        alpha_ch_M=1.0,
        alpha_ch_gamma=1.0,
        alpha_ch_eps=0.1,
        v_supg=1.0,
        v_supg_mode="residual",
        v_cip=1.0,
        fluid_convection="full",
    )

    dh0, forms0, _ = _build_problem(**base_kwargs)
    jac0 = _form_hashes(forms0.jacobian_form, dh0, rank=2)
    res0 = _form_hashes(forms0.residual_form, dh0, rank=1)

    varied_cases = (
        dict(dt_val=0.025),
        dict(mu_s=5.0, lambda_s=7.0),
        dict(kappa_inv=17.0),
        dict(gamma_div=0.2),
        dict(rho_f=950.0, mu_f=1.3e-3),
        dict(alpha_ch_eps=0.25),
    )

    for update in varied_cases:
        kwargs = dict(base_kwargs)
        kwargs.update(update)
        dh, forms, _ = _build_problem(**kwargs)
        assert _form_hashes(forms.jacobian_form, dh, rank=2) == jac0
        assert _form_hashes(forms.residual_form, dh, rank=1) == res0
