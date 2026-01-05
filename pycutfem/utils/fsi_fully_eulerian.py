from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from pycutfem.core.geometry import hansbo_cut_ratio
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.bitset import BitSet
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    FacetNormal,
    ElementWiseConstant,
    Pos,
    Neg,
    grad,
    inner,
    dot,
    div,
    jump,
    Identity,
    det,
    inv,
    trace,
)
from pycutfem.ufl.measures import dx, dGhost, dInterface


def _copy_bitset(bs: BitSet) -> BitSet:
    return BitSet(np.array(bs.mask, dtype=bool))


def _aligned_cut_mask(mesh: Mesh) -> np.ndarray:
    """Flag cut elements whose interface lies exactly on an element edge."""
    try:
        interface_edges = mesh.edge_bitset("interface").to_indices()
    except Exception:
        interface_edges = np.array([], dtype=int)
    if interface_edges.size == 0:
        return np.zeros(len(mesh.elements_list), dtype=bool)
    interface_set = set(map(int, interface_edges))
    aligned = np.zeros(len(mesh.elements_list), dtype=bool)
    for elem in mesh.elements_list:
        if getattr(elem, "tag", "") != "cut":
            continue
        for side_edges in elem.edges_by_side:
            for gid in side_edges:
                if int(gid) in interface_set:
                    aligned[int(elem.id)] = True
                    break
            if aligned[int(elem.id)]:
                break
    return aligned


def make_domain_sets(mesh: Mesh, *, use_aligned_interface: bool = False) -> Dict[str, BitSet]:
    fluid = mesh.element_bitset("outside")
    solid = mesh.element_bitset("inside")
    cut = mesh.element_bitset("cut")
    if use_aligned_interface and mesh.edge_bitset("interface").cardinality() > 0:
        aligned_cut = _aligned_cut_mask(mesh)
        cut_interface = BitSet(cut.mask & ~aligned_cut)
    else:
        cut_interface = _copy_bitset(cut)
    fluid_ifc = fluid | cut
    solid_ifc = solid | cut
    has_pos = fluid | cut
    has_neg = solid | cut

    ghost_both = mesh.edge_bitset("ghost_both")
    solid_ghost = mesh.edge_bitset("ghost_neg") | ghost_both
    fluid_ghost = mesh.edge_bitset("ghost_pos") | ghost_both
    return {
        "fluid_domain": _copy_bitset(fluid),
        "solid_domain": _copy_bitset(solid),
        "cut_domain": _copy_bitset(cut),
        "cut_interface": _copy_bitset(cut_interface),
        "fluid_interface": _copy_bitset(fluid_ifc),
        "solid_interface": _copy_bitset(solid_ifc),
        "has_pos": _copy_bitset(has_pos),
        "has_neg": _copy_bitset(has_neg),
        "solid_ghost": _copy_bitset(solid_ghost),
        "fluid_ghost": _copy_bitset(fluid_ghost),
    }


def build_measures(mesh: Mesh, level_set, domains: Dict[str, BitSet], qvol: int = 6):
    dx_fluid = dx(
        defined_on=domains["fluid_interface"],
        level_set=level_set,
        metadata={"q": qvol, "side": "+"},
    )
    dx_solid = dx(
        defined_on=domains["solid_interface"],
        level_set=level_set,
        metadata={"q": qvol, "side": "-"},
    )
    dGamma = dInterface(
        defined_on=domains["cut_interface"],
        level_set=level_set,
        metadata={"q": qvol + 2, "derivs": {(0, 0), (0, 1), (1, 0)}},
    )
    dG_fluid = dGhost(
        defined_on=domains["fluid_ghost"],
        level_set=level_set,
        metadata={"q": qvol, "derivs": {(0, 1), (1, 0)}},
    )
    dG_solid = dGhost(
        defined_on=domains["solid_ghost"],
        level_set=level_set,
        metadata={"q": qvol, "derivs": {(0, 1), (1, 0)}},
    )
    return dx_fluid, dx_solid, dGamma, dG_fluid, dG_solid


def hansbo_kappa(
    mesh: Mesh,
    level_set,
    *,
    theta_min: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray]:
    theta_pos_vals = np.clip(hansbo_cut_ratio(mesh, level_set, side="+"), theta_min, 1.0)
    theta_neg_vals = np.clip(hansbo_cut_ratio(mesh, level_set, side="-"), theta_min, 1.0)
    return theta_pos_vals, theta_neg_vals


def retag_inactive(
    dh: DofHandler,
    mesh: Mesh,
    *,
    theta_neg: Optional[np.ndarray] = None,
    solid_cut_drop: float = 0.0,
) -> None:
    dh.dof_tags["inactive"] = set()
    dh.tag_dofs_from_element_bitset("inactive", "u_pos_x", "inside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "u_pos_y", "inside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "p_pos_", "inside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "vs_neg_x", "outside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "vs_neg_y", "outside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "d_neg_x", "outside", strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "d_neg_y", "outside", strict=True)
    if theta_neg is not None and solid_cut_drop > 0.0:
        cut_mask = mesh.element_bitset("cut").mask
        bad = cut_mask & (theta_neg < solid_cut_drop)
        if np.any(bad):
            for field in ("vs_neg_x", "vs_neg_y", "d_neg_x", "d_neg_y"):
                dh.tag_dofs_from_element_bitset("inactive", field, bad, strict=False)


@dataclass
class FSIFormTerms:
    jacobian_form: object
    residual_form: object
    a_vol_f: object
    r_vol_f: object
    a_vol_s: object
    r_vol_s: object
    a_svc: object
    r_svc: object
    a_stab: object
    r_stab: object
    a_reg: object
    r_reg: object
    J_int: object
    R_int: object
    J_int_fluid: object
    R_int_fluid: object
    J_int_solid: object
    R_int_solid: object
    J_int_pen: object
    R_int_pen: object
    J_int_sym_fluid: Optional[object]
    R_int_sym_fluid: Optional[object]
    J_int_sym_solid: Optional[object]
    R_int_sym_solid: Optional[object]


def build_fsi_eulerian_forms(
    *,
    du_f,
    dp_f,
    du_s,
    ddisp_s,
    test_vel_f,
    test_q_f,
    test_vel_s,
    test_disp_s,
    uf_k,
    pf_k,
    uf_n,
    pf_n,
    us_k,
    us_n,
    disp_k,
    disp_n,
    dx_fluid,
    dx_solid,
    dGamma,
    dG_fluid,
    dG_solid,
    kappa_pos,
    kappa_neg,
    cell_h,
    beta_N,
    rho_f,
    rho_s,
    mu_f,
    mu_s,
    lambda_s,
    dt,
    theta,
    gamma_v,
    gamma_p,
    gamma_v_grad,
    solid_reg_eps,
    use_linear_solid: bool = True,
    solid_advect_lagged: bool = True,
    s_nitsche_value: float = 1.0,
) -> FSIFormTerms:
    I2 = Identity(2)
    n = FacetNormal()

    def epsilon_f(u):
        return 0.5 * (grad(u) + grad(u).T)

    def traction_fluid(u_vec, p_scal):
        return 2.0 * mu_f * dot(epsilon_f(u_vec), n) - p_scal * n

    def F_of(d):
        return inv(I2 - grad(d))

    def C_of(F):
        return dot(F.T, F)

    def E_of(F):
        return 0.5 * (C_of(F) - I2)

    def S_stvk(E):
        return lambda_s * trace(E) * I2 + Constant(2.0) * mu_s * E

    def sigma_s_linear(d):
        eps = 0.5 * (grad(d) + grad(d).T)
        return Constant(2.0) * mu_s * eps + lambda_s * trace(eps) * I2

    def dsigma_s_linear(delta_d):
        eps = 0.5 * (grad(delta_d) + grad(delta_d).T)
        return Constant(2.0) * mu_s * eps + lambda_s * trace(eps) * I2

    def sigma_s_nonlinear(d):
        if use_linear_solid:
            return sigma_s_linear(d)
        F = F_of(d)
        E = E_of(F)
        S = S_stvk(E)
        J = det(F)
        return (Constant(1.0) / J) * dot(dot(F, S), F.T)

    def dsigma_s(d_ref, delta_d):
        if use_linear_solid:
            return dsigma_s_linear(delta_d)
        Fk = F_of(d_ref)
        Ek = E_of(Fk)
        Sk = S_stvk(Ek)
        dF = dot(Fk, dot(grad(delta_d), Fk))
        dE = Constant(0.5) * (dot(dF.T, Fk) + dot(Fk.T, dF))
        dS = lambda_s * trace(dE) * I2 + Constant(2.0) * mu_s * dE
        Jk = det(Fk)
        dJ = Jk * trace(dot(grad(delta_d), Fk))
        term = dot(dF, dot(Sk, Fk.T)) + dot(Fk, dot(dS, Fk.T)) + dot(Fk, dot(Sk, dF.T))
        return (Constant(1.0) / Jk) * term - (dJ / Jk) * sigma_s_nonlinear(d_ref)

    def traction_solid_R(d):
        return dot(sigma_s_nonlinear(d), n)

    def traction_solid_L(delta_d, d_ref):
        return dot(dsigma_s(d_ref, delta_d), n)

    def grad_inner_jump(u, v):
        a = dot(jump(grad(u)), n)
        b = dot(jump(grad(v)), n)
        return inner(a, b)

    def g_v_f(gamma, phi_1, phi_2):
        return gamma * (cell_h * grad_inner_jump(phi_1, phi_2))

    def g_p(gamma, phi_1, phi_2):
        return gamma * (cell_h**3.0 * grad_inner_jump(phi_1, phi_2))

    def g_v_s(gamma, phi_1, phi_2):
        return gamma * (cell_h**3.0 * grad_inner_jump(phi_1, phi_2))

    def g_disp_s(gamma, phi_1, phi_2):
        return gamma * (cell_h * grad_inner_jump(phi_1, phi_2))

    jump_vel_trial = Pos(du_f) - Neg(du_s)
    jump_vel_test = Pos(test_vel_f) - Neg(test_vel_s)
    jump_vel_res = Pos(uf_k) - Neg(us_k)
    jump_test_f = Pos(test_vel_f)
    jump_test_s = Neg(test_vel_s)

    solid_adv_vel = us_n if solid_advect_lagged else us_k

    avg_flux_fluid_trial = kappa_pos * traction_fluid(Pos(du_f), Pos(dp_f))
    avg_flux_fluid_test = kappa_pos * traction_fluid(Pos(test_vel_f), -Pos(test_q_f))
    avg_flux_fluid_res = kappa_pos * traction_fluid(Pos(uf_k), Pos(pf_k))

    avg_flux_solid_trial = kappa_neg * traction_solid_L(Neg(ddisp_s), Neg(disp_k))
    avg_flux_solid_test = kappa_neg * traction_solid_L(Neg(test_vel_s), Neg(disp_k))
    avg_flux_solid_res = kappa_neg * traction_solid_R(Neg(disp_k))

    s_nitsche = Constant(s_nitsche_value)

    J_int_fluid = (-dot(avg_flux_fluid_trial, jump_test_f)) * dGamma + (dot(avg_flux_fluid_trial, jump_test_s)) * dGamma
    R_int_fluid = (-dot(avg_flux_fluid_res, jump_test_f)) * dGamma + (dot(avg_flux_fluid_res, jump_test_s)) * dGamma
    J_int_solid = (-dot(avg_flux_solid_trial, jump_test_f)) * dGamma + (dot(avg_flux_solid_trial, jump_test_s)) * dGamma
    R_int_solid = (-dot(avg_flux_solid_res, jump_test_f)) * dGamma + (dot(avg_flux_solid_res, jump_test_s)) * dGamma
    J_int_pen = (beta_N * mu_f / cell_h) * dot(jump_vel_trial, jump_vel_test) * dGamma
    R_int_pen = (beta_N * mu_f / cell_h) * dot(jump_vel_res, jump_vel_test) * dGamma

    J_int = J_int_fluid + J_int_solid + J_int_pen
    R_int = R_int_fluid + R_int_solid + R_int_pen

    J_int_sym_fluid = None
    R_int_sym_fluid = None
    J_int_sym_solid = None
    R_int_sym_solid = None
    if s_nitsche_value != 0.0:
        J_int_sym_fluid = (-s_nitsche * dot(avg_flux_fluid_test, jump_vel_trial)) * dGamma
        J_int_sym_solid = (-s_nitsche * dot(avg_flux_solid_test, jump_vel_trial)) * dGamma
        R_int_sym_fluid = (-s_nitsche * dot(avg_flux_fluid_test, jump_vel_res)) * dGamma
        R_int_sym_solid = (-s_nitsche * dot(avg_flux_solid_test, jump_vel_res)) * dGamma
        J_int = J_int + J_int_sym_fluid + J_int_sym_solid
        R_int = R_int + R_int_sym_fluid + R_int_sym_solid

    a_vol_f = (
        rho_f / dt * dot(du_f, test_vel_f)
        + theta * rho_f * dot(dot(grad(uf_k), du_f), test_vel_f)
        + theta * rho_f * dot(dot(grad(du_f), uf_k), test_vel_f)
        + theta * mu_f * inner(grad(du_f), grad(test_vel_f))
        - dp_f * div(test_vel_f)
        + test_q_f * div(du_f)
    ) * dx_fluid

    r_vol_f = (
        rho_f * dot(uf_k - uf_n, test_vel_f) / dt
        + theta * rho_f * dot(dot(grad(uf_k), uf_k), test_vel_f)
        + (Constant(1.0) - theta) * rho_f * dot(dot(grad(uf_n), uf_n), test_vel_f)
        + theta * mu_f * inner(grad(uf_k), grad(test_vel_f))
        + (Constant(1.0) - theta) * mu_f * inner(grad(uf_n), grad(test_vel_f))
        - pf_k * div(test_vel_f)
        + test_q_f * div(uf_k)
    ) * dx_fluid

    sigma_s_k = sigma_s_nonlinear(disp_k)
    sigma_s_n = sigma_s_nonlinear(disp_n)
    dsigma_s_k = dsigma_s(disp_k, ddisp_s)

    a_vol_s = (
        rho_s * dot(du_s, test_vel_s) / dt
        + theta * inner(dsigma_s_k, grad(test_vel_s))
        + (
            rho_s
            * theta
            * (
                dot(dot(grad(du_s), solid_adv_vel), test_vel_s)
                if solid_advect_lagged
                else (
                    dot(dot(grad(us_k), du_s), test_vel_s)
                    + dot(dot(grad(du_s), us_k), test_vel_s)
                )
            )
        )
    ) * dx_solid
    r_vol_s = (
        rho_s * dot(us_k - us_n, test_vel_s) / dt
        + theta * inner(sigma_s_k, grad(test_vel_s))
        + (Constant(1.0) - theta) * inner(sigma_s_n, grad(test_vel_s))
        + rho_s
        * (
            theta * dot(dot(grad(us_k), solid_adv_vel), test_vel_s)
            + (Constant(1.0) - theta) * dot(dot(grad(us_n), us_n), test_vel_s)
        )
    ) * dx_solid

    a_svc = (
        dot(ddisp_s, test_disp_s) / dt
        - theta * dot(du_s, test_disp_s)
        + (
            theta
            * (
                dot(dot(grad(ddisp_s), solid_adv_vel), test_disp_s)
                if solid_advect_lagged
                else (
                    dot(dot(grad(ddisp_s), us_k), test_disp_s)
                    + dot(dot(grad(disp_k), du_s), test_disp_s)
                )
            )
        )
    ) * dx_solid
    r_svc = (
        dot(disp_k - disp_n, test_disp_s) / dt
        - theta * dot(us_k, test_disp_s)
        - (Constant(1.0) - theta) * dot(us_n, test_disp_s)
        + theta * dot(dot(grad(disp_k), solid_adv_vel), test_disp_s)
        + (Constant(1.0) - theta) * dot(dot(grad(disp_n), us_n), test_disp_s)
    ) * dx_solid

    a_stab = (
        (Constant(2.0) * mu_f * g_v_f(gamma_v, du_f, test_vel_f) + g_p(gamma_p, dp_f, test_q_f)) * dG_fluid
        + (rho_s * g_v_s(gamma_v, du_s, test_vel_s) + Constant(2.0) * mu_s * g_disp_s(gamma_v_grad, ddisp_s, test_disp_s))
        * dG_solid
    )
    r_stab = (
        (Constant(2.0) * mu_f * g_v_f(gamma_v, uf_k, test_vel_f) + g_p(gamma_p, pf_k, test_q_f)) * dG_fluid
        + (rho_s * g_v_s(gamma_v, us_k, test_vel_s) + Constant(2.0) * mu_s * g_disp_s(gamma_v_grad, disp_k, test_disp_s))
        * dG_solid
    )

    a_reg = solid_reg_eps * (dot(du_s, test_vel_s) + dot(ddisp_s, test_disp_s)) * dx_solid
    r_reg = solid_reg_eps * (dot(us_k, test_vel_s) + dot(disp_k, test_disp_s)) * dx_solid

    jacobian_form = a_vol_f + J_int + a_vol_s + a_stab + a_svc + a_reg
    residual_form = r_vol_f + R_int + r_vol_s + r_stab + r_svc + r_reg

    return FSIFormTerms(
        jacobian_form=jacobian_form,
        residual_form=residual_form,
        a_vol_f=a_vol_f,
        r_vol_f=r_vol_f,
        a_vol_s=a_vol_s,
        r_vol_s=r_vol_s,
        a_svc=a_svc,
        r_svc=r_svc,
        a_stab=a_stab,
        r_stab=r_stab,
        a_reg=a_reg,
        r_reg=r_reg,
        J_int=J_int,
        R_int=R_int,
        J_int_fluid=J_int_fluid,
        R_int_fluid=R_int_fluid,
        J_int_solid=J_int_solid,
        R_int_solid=R_int_solid,
        J_int_pen=J_int_pen,
        R_int_pen=R_int_pen,
        J_int_sym_fluid=J_int_sym_fluid,
        R_int_sym_fluid=R_int_sym_fluid,
        J_int_sym_solid=J_int_sym_solid,
        R_int_sym_solid=R_int_sym_solid,
    )
