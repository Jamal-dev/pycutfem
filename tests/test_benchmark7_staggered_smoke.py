from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from examples.biofilms.benchmarks.seboldt import paper1_benchmark7_seboldt as mono
from examples.biofilms.benchmarks.seboldt import paper1_benchmark7_seboldt_staggered as staggered
from pycutfem.ufl.expressions import Constant


def _build_ratio_free_problem():
    problem = mono._create_problem(
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
        pressure_mean_constraint=True,
        solid_volumetric_split=True,
        full_ratio_free_state=True,
    )
    for key in ("alpha_k", "alpha_n"):
        problem[key].nodal_values[:] = 0.55
    for key in ("B_k", "B_n"):
        problem[key].nodal_values[:] = 0.45
    for key in ("p_k", "p_n", "p_pore_k", "p_pore_n"):
        problem[key].nodal_values[:] = 0.0
    for key in ("mu_k", "mu_n", "S_k", "S_n"):
        problem[key].nodal_values[:] = 0.0
    return problem


def _build_ratio_free_forms(problem):
    return mono._build_forms(
        problem,
        qdeg=6,
        dt_c=Constant(1.0e-3),
        theta=1.0,
        rho_f=1.0,
        mu_f=0.035,
        mu_b=0.035,
        mu_b_model="mu",
        kappa_inv=1.0e3,
        mu_s=1.67785e5,
        lambda_s=8.22148e6,
        phi_b=0.18,
        M_alpha=0.0,
        gamma_alpha=1.0,
        eps_alpha=0.1,
        solid_visco_eta=0.0,
        gamma_div=0.0,
        mechanics_nondim_mode="condition_balanced",
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
        gamma_phi=0.0,
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
        alpha_reg_eps_normal=0.1,
        alpha_reg_eps_tangent=0.025,
        alpha_reg_eta=1.0e-12,
        alpha_advect_with="vS",
        alpha_advection_form="advective",
        solid_model="linear",
        kappa_inv_model="refmap",
        enable_phi_evolution=True,
        include_skeleton_acceleration=True,
        rho_s0_tilde=1.1,
        skeleton_inertia_convection="full",
        storativity_c0=1.0e-3,
        fluid_convection="off",
        support_physics="stored_support",
        ds_alpha_transport=None,
        ds_B_transport=None,
        skeleton_pressure_mode="whole_domain",
        alpha_biot=None,
        drag_formulation="direct",
        reduced_support_state="alpha_B",
        full_ratio_free_state=True,
        pressure_interface_closure=True,
        pressure_interface_closure_strength=1.0,
        p_pore_fluid_gauge=True,
        p_pore_fluid_gauge_strength=1.0,
        interface_entry_closure=True,
        interface_entry_closure_strength=1.0,
        interface_entry_delta=10.0,
        interface_bjs_closure=True,
        interface_bjs_closure_strength=1.0,
        interface_bjs_gamma=1.0e3,
    )


def test_benchmark7_staggered_stage_forms_keep_interface_closures() -> None:
    problem = _build_ratio_free_problem()
    forms = _build_ratio_free_forms(problem)
    staggered._build_staggered_interface_forms(
        problem,
        qdeg=6,
        eps_alpha=0.1,
        mu_f=0.035,
        mu_s=1.67785e5,
        lambda_s=8.22148e6,
        solid_visco_eta=0.0,
        solid_model="linear",
        skeleton_pressure_mode="whole_domain",
        alpha_biot=None,
        interface_robin_l=2000.0,
    )
    stage_forms = staggered._build_stage_forms(problem, forms)

    base_flow = staggered._sum_forms(
        forms.r_momentum,
        forms.r_mass,
        getattr(forms, "r_pore", None),
        problem.get("_pressure_mean_residual_form"),
    )
    base_solid = staggered._sum_forms(forms.r_skeleton, forms.r_kinematics)

    assert len(stage_forms.flow.residual_form.integrals) >= len(base_flow.integrals)
    assert len(stage_forms.solid.residual_form.integrals) > len(base_solid.integrals)
    assert "p_pore" in stage_forms.flow.active_fields
    assert "p_pore" not in stage_forms.transport.active_fields
    assert "p_pore" not in stage_forms.solid.active_fields
    assert "u_y" in stage_forms.solid.active_fields
    assert "B" in stage_forms.transport.active_fields


def test_benchmark7_interface_corner_taper_is_wall_zero_and_interior_one() -> None:
    problem = _build_ratio_free_problem()
    meta = mono._configure_interface_corner_taper(problem=problem, Lx=1.0, nx=1, eps_alpha=0.1)

    taper = problem["interface_corner_taper"]
    assert taper is not None
    assert float(meta["width"]) > 0.0

    coords = np.asarray(problem["dh"].get_dof_coords(str(taper.field_name)), dtype=float)[:, 0]
    values = np.asarray(taper.nodal_values, dtype=float)
    left_idx = int(np.argmin(coords))
    right_idx = int(np.argmax(coords))

    assert values[left_idx] == pytest.approx(0.0)
    assert values[right_idx] == pytest.approx(0.0)
    assert float(np.max(values)) == pytest.approx(1.0)


def test_benchmark7_staggered_zero_step_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "jit_cache_cpp"))
    args = staggered._parse_args(
        [
            "--backend",
            "python",
            "--linear-backend",
            "scipy",
            "--kappa-list",
            "1e-3",
            "--nx",
            "1",
            "--ny",
            "2",
            "--poly-order",
            "2",
            "--pressure-order",
            "1",
            "--scalar-order",
            "1",
            "--dt",
            "1e-4",
            "--t-final",
            "0",
            "--vtk-every",
            "0",
            "--profile-samples",
            "8",
            "--outdir",
            str(tmp_path / "out"),
        ]
    )
    result = staggered._run_case(args, kappa=1.0e-3, outdir=tmp_path / "out" / "kappa_1e-3")

    assert result.summary_row["split_scheme"] == "solid_flow_transport"
    assert result.summary_row["solve_completed"] == 1.0
    assert "interface_line_point_count" in result.summary_row
    assert (tmp_path / "out" / "kappa_1e-3" / "summary.json").exists()
    assert (tmp_path / "out" / "kappa_1e-3" / "interface_diagnostics_summary.json").exists()
    assert (tmp_path / "out" / "kappa_1e-3" / "profile_final.csv").exists()
    assert (tmp_path / "out" / "kappa_1e-3" / "final_state.vtu").exists()
