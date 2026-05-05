"""Physical Stoter-style fixed-bed driver for the three-constituent model."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from examples.biofilms.benchmarks.stoter.paper1_benchmark_stoter_channel_porous import (
    Geometry,
    _eval_scalar_at_point,
    _eval_scalar_grad_at_point,
    _stokes_phase_fields,
)
from examples.biofilms.benchmarks.three_constituent.seboldt_physical import (
    _field_stats,
    _make_homogeneous_bcs,
    _make_spaces,
    _make_state,
    _make_trial_test,
    _tag_inactive_three_constituent_domains,
    _zero_scalar,
)
from examples.utils.biofilm.three_constituent_one_domain import (
    ThreeConstituentOneDomainForms,
    _named_c,
    build_three_constituent_one_domain_forms,
    build_three_constituent_pdas_solver,
)
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, TimeStepperParameters, VIParameters
from pycutfem.ufl.autodiff import linearize_form
from pycutfem.ufl.expressions import Function, div, dot, grad, inner
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad


@dataclass(frozen=True)
class PhysicalStoterResult:
    passed: bool
    outdir: Path
    summary: dict[str, object]
    centerline_rows: list[dict[str, float]]


def _tag_stoter_boundaries(mesh: Mesh, geom: Geometry, tol: float = 1.0e-12) -> None:
    cx = float(geom.center_x)
    rin = float(geom.r_in)
    mesh.tag_boundary_edges(
        {
            "bottom_active": lambda x, y: abs(y - 0.0) <= tol and abs(x - cx) <= rin + tol,
            "bottom_rest": lambda x, y: abs(y - 0.0) <= tol and not (abs(x - cx) <= rin + tol),
            "top_active": lambda x, y: abs(y - float(geom.Ly)) <= tol and abs(x - cx) <= rin + tol,
            "top_rest": lambda x, y: abs(y - float(geom.Ly)) <= tol and not (abs(x - cx) <= rin + tol),
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - float(geom.Lx)) <= tol,
        }
    )


def _bottom_inlet(x: float, *, geom: Geometry, u_max: float, ramp: float = 1.0) -> float:
    xi = (float(x) - (float(geom.center_x) - float(geom.r_in))) / max(2.0 * float(geom.r_in), 1.0e-30)
    if xi < 0.0 or xi > 1.0:
        return 0.0
    return float(ramp) * 4.0 * float(u_max) * xi * (1.0 - xi)


def _set_scalar_values(dh: DofHandler, fun: Function, fn) -> None:
    field = str(fun.field_name)
    gds = np.asarray(dh.get_field_slice(field), dtype=int)
    coords = np.asarray(dh.get_dof_coords(field), dtype=float)
    vals = np.asarray([fn(float(x), float(y)) for x, y in coords], dtype=float)
    fun.set_nodal_values(gds, vals)


def _write_centerline(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sample_centerline(
    *,
    dh: DofHandler,
    mesh: Mesh,
    geom: Geometry,
    state: dict[str, object],
    n_samples: int,
    hydraulic_conductivity: float | None = None,
) -> list[dict[str, float]]:
    x = float(geom.center_x)
    ys = np.linspace(0.0, float(geom.Ly), int(n_samples), dtype=float)
    rows: list[dict[str, float]] = []
    K = None if hydraulic_conductivity is None else float(hydraulic_conductivity)
    for y in ys:
        pt = (x, float(y))
        alpha = _eval_scalar_at_point(dh, mesh, "alpha", state["alpha_k"], pt)
        phi = _eval_scalar_at_point(dh, mesh, "phi", state["phi_k"], pt)
        F = 1.0 - alpha
        P = alpha * phi
        B = alpha * (1.0 - phi)
        vf_x = _eval_scalar_at_point(dh, mesh, "vf_x", state["v_f_k"].components[0], pt)
        vf_y = _eval_scalar_at_point(dh, mesh, "vf_y", state["v_f_k"].components[1], pt)
        vp_x = _eval_scalar_at_point(dh, mesh, "vp_x", state["v_p_k"].components[0], pt)
        vp_y = _eval_scalar_at_point(dh, mesh, "vp_y", state["v_p_k"].components[1], pt)
        if K is None:
            vp_proj_x = float("nan")
            vp_proj_y = float("nan")
        else:
            grad_pp = np.asarray(_eval_scalar_grad_at_point(dh, mesh, "pp", state["p_p_k"], pt), dtype=float)
            vp_proj_x = -K * float(grad_pp[0])
            vp_proj_y = -K * float(grad_pp[1])
        rows.append(
            {
                "x_mm": x,
                "y_mm": float(y),
                "alpha": alpha,
                "phi": phi,
                "F": F,
                "P": P,
                "B": B,
                "vf_x": vf_x,
                "vf_y": vf_y,
                "vp_x": vp_x,
                "vp_y": vp_y,
                "vp_proj_x": vp_proj_x,
                "vp_proj_y": vp_proj_y,
                "vp_proj_error_y": vp_y - vp_proj_y,
                "pf": _eval_scalar_at_point(dh, mesh, "pf", state["p_f_k"], pt),
                "pp": _eval_scalar_at_point(dh, mesh, "pp", state["p_p_k"], pt),
                "Gamma": _eval_scalar_at_point(dh, mesh, "Gamma", state["Gamma_k"], pt),
                "v_mix_y": F * vf_y + P * vp_y,
            }
        )
    return rows


def _write_summary_csv(path: Path, summary: dict[str, object]) -> None:
    flat = {
        key: value
        for key, value in summary.items()
        if isinstance(value, (int, float, str, bool)) or value is None
    }
    metrics = summary.get("metrics")
    if isinstance(metrics, dict):
        flat.update({str(key): value for key, value in metrics.items() if isinstance(value, (int, float, str, bool))})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)


def _with_stoter_bjs_interface_law(
    forms,
    *,
    state: dict[str, object],
    trial: dict[str, object],
    test: dict[str, object],
    measure,
    bjs_coefficient: float,
    delta_epsilon: float,
):
    """Add the Stoter diffuse BJS tangential law to free-fluid momentum."""

    coeff = _named_c("tc_stoter_bjs_coefficient", float(bjs_coefficient))
    if abs(float(bjs_coefficient)) == 0.0:
        return forms, coeff

    eps = max(float(delta_epsilon), 1.0e-30)
    eps_c = _named_c("tc_stoter_bjs_delta_epsilon", eps)
    eps_sq = _named_c("tc_stoter_bjs_delta_epsilon_sq", eps * eps)
    sqrt_half = _named_c("tc_stoter_bjs_sqrt_half", 0.5)

    g_alpha = grad(state["alpha_n"])
    mag_sq = dot(g_alpha, g_alpha) + eps_sq
    abs_grad_alpha = mag_sq ** sqrt_half
    delta_gamma = abs_grad_alpha - eps_c

    v = state["v_f_k"]
    w = test["w_f"]
    tangential_inner = dot(v, w) - (dot(v, g_alpha) * dot(w, g_alpha)) / mag_sq
    r_bjs = coeff * tangential_inner * delta_gamma * measure

    residual_form = forms.residual_form + r_bjs
    coefficients = [
        state["v_f_k"],
        state["p_f_k"],
        state["v_p_k"],
        state["p_p_k"],
        state["v_s_k"],
        state["u_s_k"],
        state["alpha_k"],
        state["phi_k"],
        state["Gamma_k"],
    ]
    directions = [
        trial["dv_f"],
        trial["dp_f"],
        trial["dv_p"],
        trial["dp_p"],
        trial["dv_s"],
        trial["du_s"],
        trial["dalpha"],
        trial["dphi"],
        trial["dGamma"],
    ]
    jacobian_form = linearize_form(residual_form, coefficients, directions)
    return (
        replace(
            forms,
            residual_form=residual_form,
            jacobian_form=jacobian_form,
            r_momentum_f=forms.r_momentum_f + r_bjs,
            r_total_momentum=forms.r_total_momentum + r_bjs,
            r_momentum_terms={**forms.r_momentum_terms, "free_stoter_bjs": r_bjs},
            a_terms={**forms.a_terms, "total": jacobian_form},
        ),
        coeff,
    )


def _eps(v):
    return _named_c("tc_stoter_eps_half", 0.5) * (grad(v) + grad(v).T)


def _build_stoter_mixed_limit_forms(
    *,
    state: dict[str, object],
    trial: dict[str, object],
    test: dict[str, object],
    measure,
    hydraulic_conductivity: float,
    viscosity: float,
    density: float,
    gravity_g: float,
    friction_alpha: float,
    fictitious_kappa: float,
    lsic_tau: float,
    phi0: float,
    include_convection: bool,
    dim: int = 2,
) -> ThreeConstituentOneDomainForms:
    """Build the rigid Stoter Stokes-Darcy limit in the three-field layout.

    The active physics are the Stoter diffuse-interface equations.  The pore
    velocity is retained as an unknown through a weak projection
    ``v_p = -K grad(p_p)`` so the output stays in the three-velocity notation.
    Solid velocity, displacement, alpha, phi, and Gamma are fixed by identity
    rows, which is the algebraic form of deleting those rows in the rigid-bed
    limit while keeping the same mixed element.
    """

    v_f = state["v_f_k"]
    p_f = state["p_f_k"]
    v_p = state["v_p_k"]
    p_p = state["p_p_k"]
    v_s = state["v_s_k"]
    u_s = state["u_s_k"]
    alpha = state["alpha_k"]
    alpha_n = state["alpha_n"]
    phi = state["phi_k"]
    Gamma = state["Gamma_k"]

    w_f = test["w_f"]
    q_f = test["q_f"]
    w_p = test["w_p"]
    q_p = test["q_p"]
    w_s = test["w_s"]
    z_u = test["z_u"]
    z_alpha = test["z_alpha"]
    q_s = test["q_s"]
    z_Gamma = test["z_Gamma"]

    one = _named_c("tc_stoter_limit_one", 1.0)
    zero = _named_c("tc_stoter_limit_zero", 0.0)
    minus_one = _named_c("tc_stoter_limit_minus_one", -1.0)
    half = _named_c("tc_stoter_limit_half", 0.5)
    K = _named_c("tc_stoter_hydraulic_conductivity", float(hydraulic_conductivity))
    nu = _named_c("tc_stoter_viscosity", float(viscosity))
    rho = _named_c("tc_stoter_density", float(density))
    g = _named_c("tc_stoter_gravity_g", float(gravity_g))
    kappa_floor = _named_c("tc_stoter_fictitious_kappa", float(fictitious_kappa))
    lsic = _named_c("tc_stoter_lsic_tau", float(lsic_tau))
    phi_ref = _named_c("tc_stoter_phi_ref", float(phi0))
    conv_factor = _named_c("tc_stoter_conv_factor", 1.0 if bool(include_convection) else 0.0)

    c_stokes = one + minus_one * alpha_n
    c_darcy = alpha_n * phi
    c_mod = (one + minus_one * kappa_floor) * c_stokes + kappa_floor
    c_d_mod = (one + minus_one * kappa_floor) * c_darcy + kappa_floor

    grad_c = minus_one * grad(alpha_n)
    abs_grad_c = (inner(grad_c, grad_c) + _named_c("tc_stoter_abs_grad_eps_sq", 1.0e-16)) ** half
    tau_x = minus_one * grad_c[1] / abs_grad_c
    tau_y = grad_c[0] / abs_grad_c
    u_tang = v_f[0] * tau_x + v_f[1] * tau_y
    w_tang = w_f[0] * tau_x + w_f[1] * tau_y
    bjs_coeff = _named_c(
        "tc_stoter_reference_bjs_coefficient",
        float(friction_alpha) / math.sqrt(max(float(hydraulic_conductivity), 1.0e-30)),
    )

    conv = dot(dot(grad(v_f), v_f), w_f)
    r_free_momentum = (
        conv_factor * rho * conv * c_mod
        + _named_c("tc_stoter_two", 2.0) * nu * inner(_eps(v_f), _eps(w_f)) * c_mod
        - p_f * div(w_f) * c_mod
        + lsic * div(v_f) * div(w_f) * c_mod
        - g * p_p * dot(w_f, grad_c)
        + bjs_coeff * u_tang * w_tang * abs_grad_c
    ) * measure
    r_free_mass = q_f * div(v_f) * c_mod * measure

    # Scalar Darcy row from Stoter's weak form, plus a velocity projection row
    # for reporting the pore velocity in the three-velocity model variables.
    r_pore_mass = (K * inner(grad(q_p), grad(p_p)) * c_d_mod + q_p * dot(v_f, grad_c)) * measure
    r_pore_projection = dot(v_p + K * grad(p_p), w_p) * c_d_mod * measure

    r_solid_velocity = dot(v_s, w_s) * measure
    r_solid_displacement = dot(u_s, z_u) * measure
    r_alpha = z_alpha * (alpha - alpha_n) * measure
    r_phi = q_s * (phi - phi_ref) * measure
    r_gamma = z_Gamma * (Gamma - zero) * measure

    residual_form = (
        r_alpha
        + r_free_mass
        + r_pore_mass
        + r_free_momentum
        + r_pore_projection
        + r_solid_velocity
        + r_solid_displacement
        + r_phi
        + r_gamma
    )

    coefficients = [
        state["v_f_k"],
        state["p_f_k"],
        state["v_p_k"],
        state["p_p_k"],
        state["v_s_k"],
        state["u_s_k"],
        state["alpha_k"],
        state["phi_k"],
        state["Gamma_k"],
    ]
    directions = [
        trial["dv_f"],
        trial["dp_f"],
        trial["dv_p"],
        trial["dp_p"],
        trial["dv_s"],
        trial["du_s"],
        trial["dalpha"],
        trial["dphi"],
        trial["dGamma"],
    ]
    jacobian_form = linearize_form(residual_form, coefficients, directions)

    r_mass_s = r_phi
    r_momentum_s = r_solid_velocity
    r_kinematics = r_solid_displacement
    return ThreeConstituentOneDomainForms(
        jacobian_form=jacobian_form,
        residual_form=residual_form,
        r_alpha=r_alpha,
        r_mass_f=r_free_mass,
        r_mass_p=r_pore_mass,
        r_mass_s=r_mass_s,
        r_momentum_f=r_free_momentum,
        r_momentum_p=r_pore_projection,
        r_momentum_s=r_momentum_s,
        r_kinematics=r_kinematics,
        r_gamma=r_gamma,
        r_inactive_extension=zero * r_alpha,
        r_total_mass=r_free_mass + r_pore_mass + r_mass_s,
        r_total_momentum=r_free_momentum + r_pore_projection + r_momentum_s,
        r_mass_terms={
            "free": r_free_mass,
            "pore_darcy_scalar": r_pore_mass,
            "solid_phi_identity": r_mass_s,
            "gamma_identity": r_gamma,
        },
        r_momentum_terms={
            "free_stokes_darcy": r_free_momentum,
            "pore_velocity_projection": r_pore_projection,
            "solid_velocity_identity": r_momentum_s,
        },
        r_internal_force_terms={
            "stoter_grad_c": grad_c,
            "stoter_abs_grad_c": abs_grad_c,
            "stoter_c_mod": c_mod,
            "stoter_c_d_mod": c_d_mod,
        },
        r_kinematics_terms={
            "solid_displacement_identity": r_kinematics,
            "alpha_identity": r_alpha,
            "phi_identity": r_phi,
        },
        a_terms={"total": jacobian_form},
    )


def run_physical_stoter_three_constituent(
    *,
    outdir: Path,
    nx: int = 16,
    ny: int = 20,
    eps: float = 5.0,
    phi0: float = 1.0,
    u_max: float = 1.0,
    dt: float = 5.0e-3,
    final_time: float = 5.0e-3,
    poly_order: int = 1,
    pressure_order: int = 1,
    scalar_order: int = 1,
    rho_f: float = 1.0,
    rho_p: float = 1.0,
    rho_s: float = 1.0,
    mu_f: float = 2.927,
    mu_p: float = 2.927,
    mu_s: float = 1.0e8,
    lambda_s: float = 1.0e8,
    kappa: float = 1.0,
    friction_alpha: float = 1.0,
    hydraulic_conductivity: float | None = None,
    gravity_g: float = 9.81,
    fictitious_kappa: float = 1.0e-4,
    lsic_scale: float = 1.0,
    bjs_factor: float = 1.0,
    bjs_delta_epsilon: float = 1.0e-12,
    ell_gamma_factor: float = 1.0,
    gamma_mobility: str = "interface_delta",
    resistance_model: str = "full_cholesky",
    formulation: str = "stoter_mixed_limit",
    include_convection: bool = False,
    backend: str = "cpp",
    linear_backend: str = "scipy",
    quad_order: int = 6,
    max_newton_iter: int = 12,
    newton_tol: float = 1.0e-7,
    pdas_c: float = 1.0,
    inactive_alpha_low: float = 0.02,
    inactive_alpha_high: float = 0.98,
    pore_pressure_lower_bound: float | None = None,
    centerline_samples: int = 201,
) -> PhysicalStoterResult:
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    geom = Geometry()
    h = min(float(geom.Lx) / float(nx), float(geom.Ly) / float(ny))

    nodes, elems, _, corners = structured_quad(float(geom.Lx), float(geom.Ly), nx=int(nx), ny=int(ny), poly_order=int(poly_order))
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=int(poly_order),
    )
    _tag_stoter_boundaries(mesh, geom)

    me = MixedElement(
        mesh,
        field_specs={
            "vf_x": int(poly_order),
            "vf_y": int(poly_order),
            "pf": int(pressure_order),
            "vp_x": int(poly_order),
            "vp_y": int(poly_order),
            "pp": int(pressure_order),
            "vs_x": int(poly_order),
            "vs_y": int(poly_order),
            "us_x": int(poly_order),
            "us_y": int(poly_order),
            "alpha": int(scalar_order),
            "phi": int(scalar_order),
            "Gamma": int(scalar_order),
        },
    )
    dh = DofHandler(me, method="cg")
    tol_pin = 0.51 * float(h)
    dh.tag_dof_by_locator(
        "pressure_pin",
        "pf",
        lambda x, y: abs(float(x) - float(geom.center_x)) <= tol_pin and abs(float(y) - 0.0) <= tol_pin,
    )
    spaces = _make_spaces(dh)
    trial, test = _make_trial_test(dh, spaces)
    state = _make_state(dh)

    c_stokes, _, _, _, _ = _stokes_phase_fields(geom, float(eps))
    alpha_fn = lambda x, y: float(np.clip(1.0 - float(c_stokes(x, y)), 0.0, 1.0))
    phi_fn = lambda x, y: float(phi0)
    zero_v = lambda x, y: np.asarray([0.0, 0.0], dtype=float)
    inlet_v = lambda x, y: np.asarray([0.0, _bottom_inlet(x, geom=geom, u_max=u_max)], dtype=float)
    for key in ("v_f_k", "v_f_n"):
        state[key].set_values_from_function(inlet_v if key == "v_f_k" else zero_v)
    for key in ("v_p_k", "v_p_n", "v_s_k", "v_s_n", "u_s_k", "u_s_n"):
        state[key].set_values_from_function(zero_v)
    for key in ("p_f_k", "p_p_k", "Gamma_k", "Gamma_n"):
        state[key].nodal_values[:] = 0.0
    for key in ("alpha_k", "alpha_n"):
        _set_scalar_values(dh, state[key], alpha_fn)
    for key in ("phi_k", "phi_n"):
        _set_scalar_values(dh, state[key], phi_fn)

    stoter_dx = dx(metadata={"q": int(quad_order)})
    K_bjs = float(kappa) if hydraulic_conductivity is None else float(hydraulic_conductivity)
    bjs_coefficient = float(bjs_factor) * float(friction_alpha) / math.sqrt(max(K_bjs, 1.0e-30))
    formulation_key = str(formulation).strip().lower().replace("-", "_")
    use_stoter_limit = formulation_key in {"stoter_mixed_limit", "mixed_limit", "stoter_limit", "reference_limit"}
    if use_stoter_limit:
        R_ps = float(mu_f) / max(float(kappa), 1.0e-30)
        R_fp = R_ps
        R_fs = R_ps
        ell_gamma = 0.0
        forms = _build_stoter_mixed_limit_forms(
            state=state,
            trial=trial,
            test=test,
            measure=stoter_dx,
            hydraulic_conductivity=float(K_bjs),
            viscosity=float(mu_f),
            density=float(rho_f),
            gravity_g=float(gravity_g),
            friction_alpha=float(bjs_factor) * float(friction_alpha),
            fictitious_kappa=float(fictitious_kappa),
            lsic_tau=float(lsic_scale) * float(u_max) * float(h) / 2.0,
            phi0=float(phi0),
            include_convection=bool(include_convection),
        )
        bjs_coeff_expr = _named_c("tc_stoter_reference_bjs_coefficient", bjs_coefficient)
        inactive_counts = {"stoter_limit": 0}
        rigid_fields: tuple[str, ...] = ()
    else:
        dt_c = _named_c("tc_stoter_dt", float(dt))
        R_ps = float(mu_f) / max(float(kappa), 1.0e-30)
        R_fp = R_ps
        R_fs = R_ps
        ell_gamma = float(ell_gamma_factor) * R_ps
        R_pair_cholesky = None
        if str(resistance_model).strip().lower() in {"full", "full_cholesky", "cholesky", "spd"}:
            R_pair_cholesky = (
                (math.sqrt(max(R_fp, 0.0)), 0.0, 0.0),
                (0.0, math.sqrt(max(R_fs, 0.0)), 0.0),
                (0.0, 0.0, math.sqrt(max(R_ps, 0.0))),
            )

        forms = build_three_constituent_one_domain_forms(
            **state,
            **trial,
            **test,
            dx=stoter_dx,
            dt=dt_c,
            rho_f=rho_f,
            rho_p=rho_p,
            rho_s=rho_s,
            mu_f=mu_f,
            mu_p=mu_p,
            mu_s=mu_s,
            lambda_s=lambda_s,
            R_fp=R_fp,
            R_fs=R_fs,
            R_ps=R_ps,
            R_pair_cholesky=R_pair_cholesky,
            pair_weight_epsilon=1.0e-12 if R_pair_cholesky is not None else 0.0,
            theta_fp=0.5,
            ell_Gamma=ell_gamma,
            gamma_mobility=gamma_mobility,
            transfer_velocity="free",
            lag_alpha_in_constitutive_laws=True,
            inactive_velocity_extension_factor=0.0,
            inactive_pressure_extension_factor=0.0,
            inactive_phi_extension_factor=0.0,
            inactive_displacement_extension_factor=0.0,
            phi_extension_value=phi0,
        )
        forms, bjs_coeff_expr = _with_stoter_bjs_interface_law(
            forms,
            state=state,
            trial=trial,
            test=test,
            measure=stoter_dx,
            bjs_coefficient=bjs_coefficient,
            delta_epsilon=float(bjs_delta_epsilon),
        )

        inactive_counts, _ = _tag_inactive_three_constituent_domains(
            dh,
            mesh,
            state["alpha_n"],
            alpha_low=float(inactive_alpha_low),
            alpha_high=float(inactive_alpha_high),
            previous_tagged=None,
        )
        inactive = set(int(d) for d in getattr(dh, "dof_tags", {}).get("inactive", set()) or set())
        rigid_fields = ("vs_x", "vs_y", "us_x", "us_y")
        for field in rigid_fields:
            inactive.update(int(d) for d in np.asarray(dh.get_field_slice(field), dtype=int).tolist())
        dh.dof_tags["inactive"] = inactive

    alpha_bc = lambda x, y, t: alpha_fn(x, y)
    phi_bc = lambda x, y, t: float(phi0)
    inlet_bc = lambda x, y, t: _bottom_inlet(x, geom=geom, u_max=u_max)
    bcs: list[BoundaryCondition] = []
    if use_stoter_limit:
        bcs.extend(
            [
                BoundaryCondition("vf_x", "dirichlet", "bottom_active", _zero_scalar),
                BoundaryCondition("vf_y", "dirichlet", "bottom_active", inlet_bc),
                BoundaryCondition("vf_x", "dirichlet", "bottom_rest", _zero_scalar),
                BoundaryCondition("vf_y", "dirichlet", "bottom_rest", _zero_scalar),
                BoundaryCondition("pf", "dirichlet", "pressure_pin", _zero_scalar),
            ]
        )
    else:
        for tag in ("left", "right", "bottom_rest", "top_rest"):
            bcs.extend(
                [
                    BoundaryCondition("vf_x", "dirichlet", tag, _zero_scalar),
                    BoundaryCondition("vf_y", "dirichlet", tag, _zero_scalar),
                    BoundaryCondition("vp_x", "dirichlet", tag, _zero_scalar),
                    BoundaryCondition("vp_y", "dirichlet", tag, _zero_scalar),
                    BoundaryCondition("pp", "dirichlet", tag, _zero_scalar),
                    BoundaryCondition("alpha", "dirichlet", tag, alpha_bc),
                    BoundaryCondition("phi", "dirichlet", tag, phi_bc),
                ]
            )
        bcs.extend(
            [
                BoundaryCondition("vf_x", "dirichlet", "bottom_active", _zero_scalar),
                BoundaryCondition("vf_y", "dirichlet", "bottom_active", inlet_bc),
                BoundaryCondition("vp_x", "dirichlet", "bottom_active", _zero_scalar),
                BoundaryCondition("vp_y", "dirichlet", "bottom_active", _zero_scalar),
                BoundaryCondition("pf", "dirichlet", "top_active", _zero_scalar),
                BoundaryCondition("pp", "dirichlet", "top_active", _zero_scalar),
                BoundaryCondition("alpha", "dirichlet", "bottom_active", alpha_bc),
                BoundaryCondition("alpha", "dirichlet", "top_active", alpha_bc),
                BoundaryCondition("phi", "dirichlet", "bottom_active", phi_bc),
                BoundaryCondition("phi", "dirichlet", "top_active", phi_bc),
            ]
        )
    bcs_homog = _make_homogeneous_bcs(bcs)
    pp_lo = None if pore_pressure_lower_bound is None else float(pore_pressure_lower_bound)

    solver = build_three_constituent_pdas_solver(
        forms,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(
            newton_tol=float(newton_tol),
            newton_rtol=0.0,
            max_newton_iter=int(max_newton_iter),
            print_level=1,
            line_search=True,
            ls_max_iter=16,
        ),
        vi_params=VIParameters(
            c=float(pdas_c),
            c_by_field={"alpha": float(pdas_c), "phi": float(pdas_c), "pp": float(pdas_c)},
            project_initial_guess=True,
            project_each_iteration=False,
            active_set_persistence=1,
            inactive_reg_lambda0=1.0e-10,
            inactive_reg_lambda_max=1.0e4,
        ),
        lin_params=LinearSolverParameters(backend=str(linear_backend), tol=1.0e-10, maxit=10000),
        backend=str(backend),
        quad_order=int(quad_order),
        alpha_bounds=(0.0, 1.0),
        phi_bounds=(0.0, 1.0),
        pore_pressure_bounds=(pp_lo, None),
    )

    functions = [
        state["v_f_k"],
        state["p_f_k"],
        state["v_p_k"],
        state["p_p_k"],
        state["v_s_k"],
        state["u_s_k"],
        state["alpha_k"],
        state["phi_k"],
        state["Gamma_k"],
    ]
    p_f_n = Function("p_f_n", "pf", dof_handler=dh)
    p_p_n = Function("p_p_n", "pp", dof_handler=dh)
    prev_functions = [
        state["v_f_n"],
        p_f_n,
        state["v_p_n"],
        p_p_n,
        state["v_s_n"],
        state["u_s_n"],
        state["alpha_n"],
        state["phi_n"],
        state["Gamma_n"],
    ]
    t0 = time.perf_counter()
    passed = False
    error = ""
    n_steps = 0
    elapsed = 0.0
    try:
        step_final_time = float(dt) if use_stoter_limit else float(final_time)
        _, n_steps, elapsed = solver.solve_time_interval(
            functions=functions,
            prev_functions=prev_functions,
            time_params=TimeStepperParameters(
                dt=float(dt),
                final_time=float(step_final_time),
                max_steps=1 if use_stoter_limit else max(1, int(math.ceil(float(final_time) / max(float(dt), 1.0e-30)))),
                stop_on_steady=False,
                allow_dt_reduction=False,
                predictor="prev",
            ),
        )
        passed = True
    except Exception as exc:  # noqa: PERF203
        elapsed = time.perf_counter() - t0
        error = f"{type(exc).__name__}: {exc}"
        for f, f_prev in zip(functions, prev_functions):
            f.nodal_values[:] = f_prev.nodal_values[:]

    metrics = _field_stats(state)
    centerline_rows = _sample_centerline(
        dh=dh,
        mesh=mesh,
        geom=geom,
        state=state,
        n_samples=int(centerline_samples),
        hydraulic_conductivity=float(K_bjs),
    )
    finite_mix = np.asarray([row["v_mix_y"] for row in centerline_rows if np.isfinite(row["v_mix_y"])], dtype=float)
    finite_gamma = np.asarray([row["Gamma"] for row in centerline_rows if np.isfinite(row["Gamma"])], dtype=float)
    finite_vp_proj_err = np.asarray(
        [row["vp_proj_error_y"] for row in centerline_rows if np.isfinite(row["vp_proj_error_y"])],
        dtype=float,
    )
    if finite_mix.size:
        metrics["centerline_v_mix_y_min"] = float(np.min(finite_mix))
        metrics["centerline_v_mix_y_max"] = float(np.max(finite_mix))
        metrics["centerline_v_mix_y_linf"] = float(np.linalg.norm(finite_mix, ord=np.inf))
    if finite_gamma.size:
        metrics["centerline_Gamma_linf"] = float(np.linalg.norm(finite_gamma, ord=np.inf))
    if finite_vp_proj_err.size:
        metrics["centerline_vp_projection_error_y_linf"] = float(np.linalg.norm(finite_vp_proj_err, ord=np.inf))

    summary = {
        "case_id": "stoter_physical_three_constituent_fixed_rigid_solid",
        "passed": bool(passed),
        "error": error,
        "n_steps": int(n_steps),
        "elapsed_s": float(elapsed),
        "mesh": {
            "nx": int(nx),
            "ny": int(ny),
            "poly_order": int(poly_order),
            "pressure_order": int(pressure_order),
            "scalar_order": int(scalar_order),
            "total_dofs": int(dh.total_dofs),
            "active_dofs": int(np.asarray(getattr(solver, "active_dofs", []), dtype=int).size),
        },
        "parameters": {
            "eps": float(eps),
            "phi0": float(phi0),
            "u_max": float(u_max),
            "dt": float(dt),
            "final_time": float(final_time),
            "h": float(h),
            "formulation": str(formulation_key),
            "stoter_limit_form": bool(use_stoter_limit),
            "kappa": float(kappa),
            "fictitious_kappa": float(fictitious_kappa),
            "gravity_g": float(gravity_g),
            "lsic_tau": float(lsic_scale) * float(u_max) * float(h) / 2.0,
            "include_convection": bool(include_convection),
            "hydraulic_conductivity_for_bjs": float(K_bjs),
            "friction_alpha": float(friction_alpha),
            "bjs_factor": float(bjs_factor),
            "bjs_coefficient": float(bjs_coefficient),
            "bjs_delta_epsilon": float(bjs_delta_epsilon),
            "R_ps": float(R_ps),
            "ell_Gamma": float(ell_gamma),
            "gamma_mobility": str(gamma_mobility),
            "resistance_model": str(resistance_model),
            "inactive_alpha_low": float(inactive_alpha_low),
            "inactive_alpha_high": float(inactive_alpha_high),
            "inactive_domain_counts": inactive_counts,
            "rigid_inactive_fields": list(rigid_fields),
        },
        "metrics": metrics,
    }
    _write_centerline(outdir / "centerline.csv", centerline_rows)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary_csv(outdir / "summary.csv", summary)
    return PhysicalStoterResult(bool(passed), outdir, summary, centerline_rows)


__all__ = ["PhysicalStoterResult", "run_physical_stoter_three_constituent"]
