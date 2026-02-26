from __future__ import annotations


def test_warner1986_one_domain_slough_window_handles_roundoff() -> None:
    # Regression test: the one-domain driver missed the (5.984,5.994)d sloughing
    # window when the time-stepper produced values like 5.983999999999999 due to
    # floating-point roundoff, so D_det_prev stayed zero and no sloughing occurred.
    from examples.biofilms.benchmarks.warner import warner1986_one_domain as wod

    L_ref_m = 1.0e-3
    eps_det = 0.006

    # Intended start time 5.984d represented slightly below (typical roundoff).
    t_start = 5.984 - 1.0e-15
    D0 = wod._detachment_coeff_case(
        case_id=4,
        t_days=t_start,
        dt_days=0.01,
        L_thickness_nondim=0.7,
        L_ref_m=L_ref_m,
        eps_det_nondim=eps_det,
        lambda_shear=750.0,
        slough_mode="exact",
        slough_drop_nondim=0.5,
    )
    assert D0 > 0.0

    # "shift" mode should *not* activate the stiff detachment sink in the window.
    D_shift = wod._detachment_coeff_case(
        case_id=4,
        t_days=t_start,
        dt_days=0.01,
        L_thickness_nondim=0.7,
        L_ref_m=L_ref_m,
        eps_det_nondim=eps_det,
        lambda_shear=750.0,
        slough_mode="shift",
        slough_drop_nondim=0.5,
    )
    assert D_shift == 0.0

    # Intended end time 5.994d represented slightly below should *not* trigger
    # detachment for the next step (window is half-open in the driver).
    t_end = 5.994 - 1.0e-15
    D1 = wod._detachment_coeff_case(
        case_id=4,
        t_days=t_end,
        dt_days=0.01,
        L_thickness_nondim=0.7,
        L_ref_m=L_ref_m,
        eps_det_nondim=eps_det,
        lambda_shear=750.0,
        slough_mode="exact",
        slough_drop_nondim=0.5,
    )
    assert D1 == 0.0


def test_warner1986_one_domain_shift_slough_truncates_and_refills() -> None:
    import numpy as np

    from examples.biofilms.benchmarks.warner import warner1986_one_domain as wod

    class _DH:
        def __init__(self, coords_by_field: dict[str, np.ndarray]):
            self._coords = coords_by_field

        def get_dof_coords(self, field: str) -> np.ndarray:
            return self._coords[field]

    class _F:
        def __init__(self, values: np.ndarray):
            self.nodal_values = np.asarray(values, dtype=float).copy()

    y = np.linspace(0.0, 1.0, 2001)
    coords = np.column_stack([np.zeros_like(y), y])
    dh = _DH({"alpha": coords, "S": coords})

    # Construct a strip profile with an interface around L≈0.7.
    alpha0 = (y <= 0.7).astype(float)
    alpha = _F(alpha0)
    S = _F(np.zeros_like(y))

    L_old = float(wod._strip_thickness_alpha_half(dh=dh, alpha=alpha))

    applied = wod._apply_case4_shift_slough_strip(
        t_days=5.994,
        applied=False,
        dh=dh,
        alpha=alpha,
        S=S,
        drop_nondim=0.5,
        eps_nondim=0.006,
        S_bulk_value=20.0,
    )
    assert applied is True

    L_new = float(wod._strip_thickness_alpha_half(dh=dh, alpha=alpha))
    # Tolerance: thickness estimate depends on y-grid interpolation.
    assert abs(L_new - (L_old - 0.5)) <= 2.0 * float(y[1] - y[0])

    # New fluid region above the shifted interface is refilled to bulk.
    mask = y >= float(L_new) - 1.0e-14
    assert np.allclose(S.nodal_values[mask], 20.0, atol=0.0, rtol=0.0)

    # Calling before the event should not apply anything.
    alpha2 = _F(alpha0)
    S2 = _F(np.zeros_like(y))
    applied2 = wod._apply_case4_shift_slough_strip(
        t_days=5.993,
        applied=False,
        dh=dh,
        alpha=alpha2,
        S=S2,
        drop_nondim=0.5,
        eps_nondim=0.006,
        S_bulk_value=20.0,
    )
    assert applied2 is False
    assert np.allclose(alpha2.nodal_values, alpha0, atol=0.0, rtol=0.0)


def test_warner1986_one_domain_case2_step_bc_is_not_smeared() -> None:
    # Regression: Case 2 has a Dirichlet step at t=6d. For θ=1 schemes, evaluating
    # the BC at t_{n+θ} can smear the discontinuity across the last pre-event step
    # (and suppress the Warner spike at exactly t=6d). The driver now treats
    # t==6d as "pre-event" for Dirichlet evaluation only.
    from examples.biofilms.benchmarks.warner import warner1986_one_domain as wod

    assert wod._S_bulk_case(case_id=2, t_days=5.999, S_high=3.0) == 3.0
    assert wod._S_bulk_case(case_id=2, t_days=6.0, S_high=3.0) == 0.0
    assert wod._S_bulk_case_for_bc(case_id=2, t_days=6.0, S_high=3.0) == 3.0
    assert wod._S_bulk_case_for_bc(case_id=2, t_days=6.1, S_high=3.0) == 0.0


def test_warner1986_one_domain_warner_stencil_matches_linear_profile() -> None:
    import numpy as np

    from examples.biofilms.benchmarks.warner import warner1986_one_domain as wod

    class _DH:
        def __init__(self, coords_by_field: dict[str, np.ndarray]):
            self._coords = coords_by_field

        def get_dof_coords(self, field: str) -> np.ndarray:
            return self._coords[field]

    class _F:
        def __init__(self, values: np.ndarray):
            self.nodal_values = np.asarray(values, dtype=float).copy()

    # Construct a strip profile S(y)=S_L*(y/L) for y<=L (linear in ζ), and constant above.
    L = 0.7
    S_L = 3.0
    y = np.linspace(0.0, 1.0, 4001)
    coords = np.column_stack([np.zeros_like(y), y])
    dh = _DH({"S": coords})
    S_vals = np.where(y <= L, S_L * (y / L), S_L)
    S = _F(S_vals)

    L_ref_m = 1.0e-3
    D_S_phys = 83.0e-6
    uL_m_per_d = 0.0

    got = wod._strip_removal_warner_stencil(
        dh=dh,
        S=S,
        L_thickness_nondim=L,
        L_ref_m=L_ref_m,
        D_S_phys=D_S_phys,
        S_L=S_L,
        uL_m_per_d=uL_m_per_d,
        npoint=15,
    )

    # For S(ζ)=S_L*ζ, dS/dζ|_{ζ=1}=S_L, so removal = D*S_L/L_phys.
    L_phys = L_ref_m * L
    expected = D_S_phys * S_L / L_phys
    assert abs(got - expected) <= 1.0e-12 * max(1.0, abs(expected))
