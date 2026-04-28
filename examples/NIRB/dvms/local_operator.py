from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.operators import LocalAssemblyOperator, LocalAssemblyResult, LocalAssemblyWorkset
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.forms import Equation, Form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.expressions import Constant, ElementWiseConstant, Function, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction, TestFunction

from .helpers import (
    _bossak_coefficients,
    _kratos_dvms_current_element_size_array,
    _kratos_dvms_current_element_size_coefficient,
    _kratos_dvms_element_size_coefficient,
)
from .state import FluidDVMSState
from .symbolics import build_fluid_dvms_kratos_split_forms, build_fluid_dvms_local_forms
from .update import _dvms_fast_p1_tri_cache, _grad_phi_phys, _scalar_locals, _update_fluid_dvms_predicted_subscale


_DVMS_ZERO = Constant(0.0)
_DVMS_ZERO_VEC = Constant(np.zeros((2,), dtype=float), dim=1)


def _normalize_backend_name(backend: str | None) -> str:
    name = str(backend or "python").strip().lower()
    if name in {"c++", "cpp"}:
        return "cpp"
    if name not in {"python", "jit", "cpp"}:
        raise ValueError(f"Unsupported DVMS local-assembly backend {backend!r}.")
    return name


def _normalize_contribution_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in {"velocity", "system", "system_condensed", "mass_lhs", "mass_stabilization"}:
        raise ValueError(f"Unsupported FluidDVMS local contribution mode {mode!r}.")
    return value


def _local_grouped_gdofs_batch(dh: DofHandler, element_ids: np.ndarray) -> np.ndarray:
    eids = np.asarray(element_ids, dtype=int).reshape(-1)
    ux_map = np.asarray(dh.element_maps["ux"], dtype=int)[eids]
    uy_map = np.asarray(dh.element_maps["uy"], dtype=int)[eids]
    p_map = np.asarray(dh.element_maps["p"], dtype=int)[eids]
    return np.concatenate([ux_map, uy_map, p_map], axis=1).astype(int, copy=False)


def _local_grouped_gdofs(dh: DofHandler, eid: int) -> np.ndarray:
    return _local_grouped_gdofs_batch(dh, np.asarray([int(eid)], dtype=int))[0]


def _fluid_local_layout_positions(full_gdofs_map: np.ndarray, fluid_gdofs_map: np.ndarray) -> np.ndarray:
    full = np.asarray(full_gdofs_map, dtype=int)
    fluid = np.asarray(fluid_gdofs_map, dtype=int)
    if full.ndim != 2 or fluid.ndim != 2:
        raise ValueError("Expected batched local dof maps with shape (n_elem, n_local_dofs).")
    if int(full.shape[0]) != int(fluid.shape[0]):
        raise ValueError("Full and fluid local dof maps must have the same batch size.")
    matches = full[:, None, :] == fluid[:, :, None]
    found = np.any(matches, axis=2)
    if not np.all(found):
        raise ValueError("Unable to locate the fluid local block within the compiled mixed-element layout.")
    return np.argmax(matches, axis=2).astype(int, copy=False)


def _compress_batch_to_fluid_block(
    dh: DofHandler,
    batch,
    *,
    fluid_gdofs_map: np.ndarray | None = None,
    positions: np.ndarray | None = None,
):
    fluid_gdofs = (
        _local_grouped_gdofs_batch(dh, np.asarray(batch.element_ids, dtype=int))
        if fluid_gdofs_map is None
        else np.asarray(fluid_gdofs_map, dtype=int)
    )
    positions_arr = (
        _fluid_local_layout_positions(np.asarray(batch.gdofs_map, dtype=int), fluid_gdofs)
        if positions is None
        else np.asarray(positions, dtype=int)
    )
    lhs = None
    rhs = None
    if batch.K_elem is not None:
        k_full = np.asarray(batch.K_elem, dtype=float)
        row_pos = positions_arr[:, :, None]
        col_pos = positions_arr[:, None, :]
        lhs = np.take_along_axis(
            np.take_along_axis(k_full, row_pos, axis=1),
            col_pos,
            axis=2,
        )
    if batch.F_elem is not None:
        rhs = np.take_along_axis(np.asarray(batch.F_elem, dtype=float), positions_arr, axis=1)
    return lhs, rhs, fluid_gdofs


def _as_form(term) -> Form:
    if isinstance(term, Form):
        return term
    return Form([term])


def _sum_forms(terms) -> Form:
    result = None
    for term in terms:
        result = _as_form(term) if result is None else result + term
    if result is None:
        raise ValueError("Cannot build an empty DVMS form.")
    return result


@dataclass(frozen=True)
class _KratosResidualizedSystemForms:
    nonviscous_velocity: Equation
    viscous_velocity: Equation
    mass_matrix: Form


def _fluid_values_full(
    dh: DofHandler,
    *,
    u: VectorFunction,
    p: Function | None = None,
) -> np.ndarray:
    values = np.zeros(int(dh.total_dofs), dtype=float)
    ux_ids = np.asarray(dh.get_field_slice(u.components[0].field_name), dtype=int)
    uy_ids = np.asarray(dh.get_field_slice(u.components[1].field_name), dtype=int)
    values[ux_ids] = np.asarray(u.components[0].nodal_values, dtype=float).reshape(-1)
    values[uy_ids] = np.asarray(u.components[1].nodal_values, dtype=float).reshape(-1)
    if p is not None:
        p_ids = np.asarray(dh.get_field_slice(p.field_name), dtype=int)
        values[p_ids] = np.asarray(p.nodal_values, dtype=float).reshape(-1)
    return values


def _fluid_bossak_acceleration_values_full(
    dh: DofHandler,
    *,
    u_k: VectorFunction,
    u_prev: VectorFunction | None,
    a_prev: VectorFunction | None,
    a_curr: VectorFunction | None,
    bossak_alpha: float,
    dt: float,
) -> np.ndarray:
    if a_prev is None:
        raise ValueError("Kratos residualized DVMS system requires a_prev.")
    bossak = _bossak_coefficients(alpha=float(bossak_alpha), dt=float(dt))
    if a_curr is None:
        if u_prev is None:
            raise ValueError("Kratos residualized DVMS system requires u_prev when a_curr is not supplied.")
        current_acceleration = (
            float(bossak["ma0"])
            * (np.asarray(u_k.nodal_values, dtype=float) - np.asarray(u_prev.nodal_values, dtype=float))
            + float(bossak["ma2"]) * np.asarray(a_prev.nodal_values, dtype=float)
        )
    else:
        current_acceleration = np.asarray(a_curr.nodal_values, dtype=float)
    scheme_acceleration = (
        (1.0 - float(bossak["alpha"])) * np.asarray(current_acceleration, dtype=float)
        + float(bossak["alpha"]) * np.asarray(a_prev.nodal_values, dtype=float)
    )
    tmp = VectorFunction(
        "dvms_bossak_scheme_accel_tmp",
        [u_k.components[0].field_name, u_k.components[1].field_name],
        dof_handler=dh,
    )
    tmp.nodal_values[:] = scheme_acceleration
    return _fluid_values_full(dh, u=tmp, p=None)


def _local_trial_test_functions(dh: DofHandler):
    v_space = FunctionSpace("FluidVelocityLocal", ["ux", "uy"], dim=1)
    du = VectorTrialFunction(space=v_space, dof_handler=dh)
    v = VectorTestFunction(space=v_space, dof_handler=dh)
    dp = TrialFunction(name="dp_local", field_name="p", dof_handler=dh)
    q = TestFunction(name="q_local", field_name="p", dof_handler=dh)
    return du, dp, v, q


def _dvms_state_coefficients(state: FluidDVMSState | None):
    if isinstance(state, FluidDVMSState) and int(state.sample_count) > 0:
        return {
            "predicted_subscale": state.coefficient("predicted_subscale_velocity"),
            "old_subscale": state.coefficient("old_subscale_velocity"),
            "momentum_projection": state.coefficient("momentum_projection"),
            "mass_projection": state.coefficient("mass_projection"),
            "old_mass_residual": state.coefficient("old_mass_residual"),
        }
    return {
        "predicted_subscale": _DVMS_ZERO_VEC,
        "old_subscale": _DVMS_ZERO_VEC,
        "momentum_projection": _DVMS_ZERO_VEC,
        "mass_projection": _DVMS_ZERO,
        "old_mass_residual": _DVMS_ZERO,
    }


def _solve_batched_square(mats: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    mats_arr = np.asarray(mats, dtype=float)
    rhs_arr = np.asarray(rhs, dtype=float)
    if int(mats_arr.shape[0]) == 0:
        return np.zeros_like(rhs_arr, dtype=float)
    try:
        if rhs_arr.ndim == mats_arr.ndim - 1 and rhs_arr.shape[-1] == mats_arr.shape[-1]:
            return np.asarray(np.linalg.solve(mats_arr, rhs_arr[..., None]), dtype=float)[..., 0]
        return np.asarray(np.linalg.solve(mats_arr, rhs_arr), dtype=float)
    except np.linalg.LinAlgError:
        out = np.zeros_like(rhs_arr, dtype=float)
        for idx in range(int(mats_arr.shape[0])):
            out[idx] = np.asarray(np.linalg.solve(mats_arr[idx], rhs_arr[idx]), dtype=float)
        return out


def _dvms_condensed_hidden_state_correction_batch(
    *,
    mesh: Mesh,
    dh: DofHandler,
    element_ids: np.ndarray,
    u_k: VectorFunction,
    u_prev: VectorFunction,
    a_prev: VectorFunction,
    p_k: Function,
    d_mesh: VectorFunction,
    d_prev: VectorFunction,
    d_prev2: VectorFunction | None = None,
    mesh_v: VectorFunction | None = None,
    mesh_v_prev: VectorFunction | None = None,
    mesh_a_prev: VectorFunction | None = None,
    state: FluidDVMSState,
    rho_f: float,
    mu_f: float,
    dt: float,
    bossak_alpha: float,
    dynamic_tau: float,
    body_force: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eids = np.asarray(element_ids, dtype=int).reshape(-1)
    if eids.size == 0:
        return (
            np.zeros((0, 0, 0), dtype=float),
            np.zeros((0, 0), dtype=float),
            np.zeros((0, 0), dtype=int),
        )
    fast = _dvms_fast_p1_tri_cache(state, dh, mesh)
    if fast is None:
        raise NotImplementedError(
            "The condensed DVMS hidden-state correction currently supports only the fast P1 triangle path."
        )

    rho_value = float(rho_f)
    mu_value = float(mu_f)
    dt_value = max(float(dt), 1.0e-14)
    # Kratos' DVMS element keeps DYNAMIC_TAU in the formulation settings, but
    # d_vms.cpp hard-codes the dynamic subscale time term with coefficient 1.
    # Preserve the argument for API compatibility while matching Kratos here.
    del dynamic_tau
    dynamic_tau_value = 1.0
    bossak = _bossak_coefficients(alpha=float(bossak_alpha), dt=dt_value)
    bossak_mass_coeff = (1.0 - float(bossak["alpha"])) * float(bossak["ma0"])
    inv_dt = 1.0 / dt_value

    basis_u = np.asarray(fast["basis"]["ux"], dtype=float)
    grad_phi_u = _grad_phi_phys(fast["grad_ref"]["ux"], np.asarray(fast["J_inv"], dtype=float)[eids])
    grad_phi_p = _grad_phi_phys(fast["grad_ref"]["p"], np.asarray(fast["J_inv"], dtype=float)[eids])
    grad_phi_mx = _grad_phi_phys(fast["grad_ref"]["mx"], np.asarray(fast["J_inv"], dtype=float)[eids])
    grad_phi_my = _grad_phi_phys(fast["grad_ref"]["my"], np.asarray(fast["J_inv"], dtype=float)[eids])
    detJ_geo = np.asarray(fast["detJ"], dtype=float)[eids]
    ref_weights = np.asarray(state.quadrature_layout.reference_weights, dtype=float).reshape(1, -1)
    h_e = _kratos_dvms_current_element_size_array(
        mesh,
        dh,
        d_mesh,
        element_ids=eids,
    )

    ux_map = np.asarray(fast["element_maps"]["ux"], dtype=int)[eids]
    uy_map = np.asarray(fast["element_maps"]["uy"], dtype=int)[eids]
    p_map = np.asarray(fast["element_maps"]["p"], dtype=int)[eids]
    mx_map = np.asarray(fast["element_maps"]["mx"], dtype=int)[eids]
    my_map = np.asarray(fast["element_maps"]["my"], dtype=int)[eids]

    ux_k = _scalar_locals(dh, "ux", u_k.components[0].nodal_values, ux_map)
    uy_k = _scalar_locals(dh, "uy", u_k.components[1].nodal_values, uy_map)
    ux_prev = _scalar_locals(dh, "ux", u_prev.components[0].nodal_values, ux_map)
    uy_prev = _scalar_locals(dh, "uy", u_prev.components[1].nodal_values, uy_map)
    ax_prev = _scalar_locals(dh, "ux", a_prev.components[0].nodal_values, ux_map)
    ay_prev = _scalar_locals(dh, "uy", a_prev.components[1].nodal_values, uy_map)
    p_vals = _scalar_locals(dh, "p", p_k.nodal_values, p_map)
    mx_k = _scalar_locals(dh, "mx", d_mesh.components[0].nodal_values, mx_map)
    my_k = _scalar_locals(dh, "my", d_mesh.components[1].nodal_values, my_map)
    mx_prev = _scalar_locals(dh, "mx", d_prev.components[0].nodal_values, mx_map)
    my_prev = _scalar_locals(dh, "my", d_prev.components[1].nodal_values, my_map)
    if mesh_v is not None:
        mx_vel = _scalar_locals(dh, "mx", mesh_v.components[0].nodal_values, mx_map)
        my_vel = _scalar_locals(dh, "my", mesh_v.components[1].nodal_values, my_map)
    else:
        mx_vel = np.zeros_like(mx_prev)
        my_vel = np.zeros_like(my_prev)
    if mesh_v_prev is not None and mesh_a_prev is not None:
        mx_vel_prev = _scalar_locals(dh, "mx", mesh_v_prev.components[0].nodal_values, mx_map)
        my_vel_prev = _scalar_locals(dh, "my", mesh_v_prev.components[1].nodal_values, my_map)
        mx_acc_prev = _scalar_locals(dh, "mx", mesh_a_prev.components[0].nodal_values, mx_map)
        my_acc_prev = _scalar_locals(dh, "my", mesh_a_prev.components[1].nodal_values, my_map)
    else:
        mx_vel_prev = np.zeros_like(mx_prev)
        my_vel_prev = np.zeros_like(my_prev)
        mx_acc_prev = np.zeros_like(mx_prev)
        my_acc_prev = np.zeros_like(my_prev)
    if d_prev2 is None:
        mx_prev2 = np.zeros_like(mx_prev)
        my_prev2 = np.zeros_like(my_prev)
    else:
        mx_prev2 = _scalar_locals(dh, "mx", d_prev2.components[0].nodal_values, mx_map)
        my_prev2 = _scalar_locals(dh, "my", d_prev2.components[1].nodal_values, my_map)

    u_k_q = np.stack(
        [
            np.einsum("el,ql->eq", ux_k, basis_u, optimize=True),
            np.einsum("el,ql->eq", uy_k, basis_u, optimize=True),
        ],
        axis=2,
    )
    u_prev_q = np.stack(
        [
            np.einsum("el,ql->eq", ux_prev, basis_u, optimize=True),
            np.einsum("el,ql->eq", uy_prev, basis_u, optimize=True),
        ],
        axis=2,
    )
    a_prev_q = np.stack(
        [
            np.einsum("el,ql->eq", ax_prev, basis_u, optimize=True),
            np.einsum("el,ql->eq", ay_prev, basis_u, optimize=True),
        ],
        axis=2,
    )
    d_mesh_q = np.stack(
        [
            np.einsum("el,ql->eq", mx_k, basis_u, optimize=True),
            np.einsum("el,ql->eq", my_k, basis_u, optimize=True),
        ],
        axis=2,
    )
    d_prev_q = np.stack(
        [
            np.einsum("el,ql->eq", mx_prev, basis_u, optimize=True),
            np.einsum("el,ql->eq", my_prev, basis_u, optimize=True),
        ],
        axis=2,
    )
    mesh_v_q = np.stack(
        [
            np.einsum("el,ql->eq", mx_vel, basis_u, optimize=True),
            np.einsum("el,ql->eq", my_vel, basis_u, optimize=True),
        ],
        axis=2,
    )
    mesh_v_prev_q = np.stack(
        [
            np.einsum("el,ql->eq", mx_vel_prev, basis_u, optimize=True),
            np.einsum("el,ql->eq", my_vel_prev, basis_u, optimize=True),
        ],
        axis=2,
    )
    mesh_a_prev_q = np.stack(
        [
            np.einsum("el,ql->eq", mx_acc_prev, basis_u, optimize=True),
            np.einsum("el,ql->eq", my_acc_prev, basis_u, optimize=True),
        ],
        axis=2,
    )
    d_prev2_q = np.stack(
        [
            np.einsum("el,ql->eq", mx_prev2, basis_u, optimize=True),
            np.einsum("el,ql->eq", my_prev2, basis_u, optimize=True),
        ],
        axis=2,
    )

    grad_u_ref = np.zeros((eids.size, 2, 2), dtype=float)
    grad_u_ref[:, 0, :] = np.einsum("el,elk->ek", ux_k, grad_phi_u, optimize=True)
    grad_u_ref[:, 1, :] = np.einsum("el,elk->ek", uy_k, grad_phi_u, optimize=True)
    grad_d_ref = np.zeros((eids.size, 2, 2), dtype=float)
    grad_d_ref[:, 0, :] = np.einsum("el,elk->ek", mx_k, grad_phi_mx, optimize=True)
    grad_d_ref[:, 1, :] = np.einsum("el,elk->ek", my_k, grad_phi_my, optimize=True)
    grad_p_ref = np.einsum("el,elk->ek", p_vals, grad_phi_p, optimize=True)

    F = np.zeros((eids.size, 2, 2), dtype=float)
    F[:, 0, 0] = 1.0 + grad_d_ref[:, 0, 0]
    F[:, 0, 1] = grad_d_ref[:, 0, 1]
    F[:, 1, 0] = grad_d_ref[:, 1, 0]
    F[:, 1, 1] = 1.0 + grad_d_ref[:, 1, 1]
    detF = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
    Finv = np.zeros_like(F)
    Finv[:, 0, 0] = F[:, 1, 1] / detF
    Finv[:, 0, 1] = -F[:, 0, 1] / detF
    Finv[:, 1, 0] = -F[:, 1, 0] / detF
    Finv[:, 1, 1] = F[:, 0, 0] / detF
    J_ale = detF

    grad_u_phys = np.einsum("eij,ejk->eik", grad_u_ref, Finv, optimize=True)
    grad_p_phys = np.einsum("eji,ej->ei", Finv, grad_p_ref, optimize=True)
    grad_phi_u_phys = grad_phi_u
    grad_phi_p_phys = grad_phi_p

    if mesh_v is not None:
        w_mesh_q = mesh_v_q
    elif mesh_v_prev is not None and mesh_a_prev is not None:
        beta = float(bossak["beta"])
        gamma = float(bossak["gamma"])
        a_mesh_q = (
            d_mesh_q
            - d_prev_q
            - dt_value * mesh_v_prev_q
            - (dt_value * dt_value) * (0.5 - beta) * mesh_a_prev_q
        ) / (beta * dt_value * dt_value)
        w_mesh_q = mesh_v_prev_q + dt_value * ((1.0 - gamma) * mesh_a_prev_q + gamma * a_mesh_q)
    else:
        w_mesh_q = (1.5 * d_mesh_q - 2.0 * d_prev_q + 0.5 * d_prev2_q) / dt_value
    resolved_conv = u_k_q - w_mesh_q
    a_curr = float(bossak["ma0"]) * (u_k_q - u_prev_q) + float(bossak["ma2"]) * a_prev_q
    a_relaxed = a_curr

    n_q = int(state.n_qp_per_element)
    predicted = np.asarray(state.predicted_subscale_velocity, dtype=float).reshape(int(state.n_elements), n_q, 2)[eids]
    old_subscale = np.asarray(state.old_subscale_velocity, dtype=float).reshape(int(state.n_elements), n_q, 2)[eids]
    momentum_projection = np.asarray(state.momentum_projection, dtype=float).reshape(int(state.n_elements), n_q, 2)[eids]

    conv = resolved_conv + predicted
    conv_speed = np.linalg.norm(conv, axis=2)
    inv_tau = (
        (8.0 * mu_value / np.maximum(h_e[:, None] * h_e[:, None], 1.0e-30))
        + rho_value * (inv_dt + 2.0 * conv_speed / np.maximum(h_e[:, None], 1.0e-30))
    ) / dynamic_tau_value
    linearization = np.broadcast_to(
        rho_value * grad_u_phys[:, None, :, :],
        (eids.size, n_q, 2, 2),
    ).copy()
    linearization[:, :, 0, 0] += inv_tau
    linearization[:, :, 1, 1] += inv_tau

    old_uss_term = rho_value * inv_dt * old_subscale
    static_residual = -(
        rho_value * a_relaxed
        + rho_value * np.einsum("eij,eqj->eqi", grad_u_phys, resolved_conv, optimize=True)
        + grad_p_phys[:, None, :]
        + momentum_projection
    ) + old_uss_term
    hidden_residual = static_residual - np.einsum("eqij,eqj->eqi", linearization, predicted, optimize=True)

    tau_one = 1.0 / np.maximum(inv_tau, 1.0e-30)
    weight = detJ_geo[:, None] * ref_weights * J_ale[:, None]
    add_velocity_source = old_uss_term - momentum_projection
    if body_force is not None:
        add_velocity_source = add_velocity_source + np.asarray(body_force, dtype=float).reshape(1, 1, 2)
    tau_res_static_conv = rho_value * np.einsum("eij,eqj->eqi", grad_u_phys, conv, optimize=True)
    tau_res_static_pres = grad_p_phys[:, None, :]
    tau_source_scalar = rho_value * np.einsum("elk,eqk->eql", grad_phi_u_phys, conv, optimize=True) - rho_value * inv_dt * basis_u[None, :, :]
    mass_action = rho_value * a_relaxed

    n_elem = int(eids.shape[0])
    C = np.zeros((n_elem, n_q, 9, 2), dtype=float)
    B = np.zeros((n_elem, n_q, 2, 9), dtype=float)

    G_rows = grad_u_phys
    Ssrc = add_velocity_source
    Aconv = tau_res_static_conv
    gp = tau_res_static_pres

    # velocity test rows
    for comp in range(2):
        row_slice = slice(comp * 3, (comp + 1) * 3)
        C[:, :, row_slice, :] += (
            weight[:, :, None, None]
            * tau_one[:, :, None, None]
            * rho_value
            * Ssrc[:, :, comp][:, :, None, None]
            * grad_phi_u_phys[:, None, :, :]
        )
        C[:, :, row_slice, :] -= (
            weight[:, :, None, None]
            * rho_value
            * basis_u[None, :, :, None]
            * G_rows[:, None, comp, :][:, :, None, :]
        )
        C[:, :, row_slice, :] -= (
            weight[:, :, None, None]
            * tau_one[:, :, None, None]
            * rho_value
            * Aconv[:, :, comp][:, :, None, None]
            * grad_phi_u_phys[:, None, :, :]
        )
        C[:, :, row_slice, :] -= (
            weight[:, :, None, None]
            * tau_one[:, :, None, None]
            * tau_source_scalar[:, :, :, None]
            * rho_value
            * G_rows[:, None, comp, :][:, :, None, :]
        )
        C[:, :, row_slice, :] -= (
            weight[:, :, None, None]
            * tau_one[:, :, None, None]
            * rho_value
            * gp[:, :, comp][:, :, None, None]
            * grad_phi_u_phys[:, None, :, :]
        )
        C[:, :, row_slice, :] -= (
            weight[:, :, None, None]
            * tau_one[:, :, None, None]
            * rho_value
            * mass_action[:, :, comp][:, :, None, None]
            * grad_phi_u_phys[:, None, :, :]
        )

    # pressure test rows
    Gt_gradq = rho_value * np.einsum("eji,elj->eli", G_rows, grad_phi_p_phys, optimize=True)
    C[:, :, 6:9, :] -= weight[:, :, None, None] * tau_one[:, :, None, None] * Gt_gradq[:, None, :, :]

    s = conv
    g_dot_s = np.einsum("elk,eqk->eql", grad_phi_u_phys, s, optimize=True)
    for local in range(3):
        phi = basis_u[:, local][None, :]
        g = grad_phi_u_phys[:, local, :]
        B[:, :, 0, local] = -rho_value * (
            bossak_mass_coeff * phi
            + g_dot_s[:, :, local]
            + phi * G_rows[:, 0, 0][:, None]
        )
        B[:, :, 1, local] = -rho_value * (phi * G_rows[:, 1, 0][:, None])
        B[:, :, 0, 3 + local] = -rho_value * (phi * G_rows[:, 0, 1][:, None])
        B[:, :, 1, 3 + local] = -rho_value * (
            bossak_mass_coeff * phi
            + g_dot_s[:, :, local]
            + phi * G_rows[:, 1, 1][:, None]
        )
        B[:, :, :, 6 + local] = -grad_phi_p_phys[:, local, :][:, None, :]

    correction_sol = _solve_batched_square(
        linearization.reshape(-1, 2, 2),
        B.reshape(-1, 2, 9),
    ).reshape(n_elem, n_q, 2, 9)
    residual_sol = _solve_batched_square(
        linearization.reshape(-1, 2, 2),
        hidden_residual.reshape(-1, 2),
    ).reshape(n_elem, n_q, 2)
    K_corr = np.einsum("eqia,eqaj->eqij", C, correction_sol, optimize=True)
    F_corr = np.einsum("eqia,eqa->eqi", C, residual_sol, optimize=True)
    K_elem = np.einsum("eq,eqij->eij", np.ones_like(weight), K_corr, optimize=True)
    F_elem = np.einsum("eq,eqi->ei", np.ones_like(weight), F_corr, optimize=True)

    return K_elem, F_elem, _local_grouped_gdofs_batch(dh, eids)


def _build_fluid_dvms_form_or_equation(
    *,
    mesh: Mesh,
    dh: DofHandler,
    u_k: VectorFunction,
    u_prev: VectorFunction | None,
    a_prev: VectorFunction | None,
    a_curr: VectorFunction | None,
    p_k: Function,
    d_mesh: VectorFunction,
    d_prev: VectorFunction,
    d_prev2: VectorFunction | None,
    mesh_v: VectorFunction | None,
    mesh_v_prev: VectorFunction | None,
    mesh_a_prev: VectorFunction | None,
    state: FluidDVMSState | None,
    rho_f: float,
    mu_f: float,
    dt: float,
    bossak_alpha: float,
    quadrature_order: int | None,
    body_force: np.ndarray | None,
    use_oss: bool,
    contribution_mode: str,
    h_coefficient: ElementWiseConstant | None = None,
):
    mode = _normalize_contribution_mode(contribution_mode)
    if mode in {"system", "system_condensed"} and (u_prev is None or a_prev is None):
        raise ValueError(f"FluidDVMS local contribution mode {mode!r} requires u_prev and a_prev.")

    du, dp, v, q = _local_trial_test_functions(dh)
    bossak = _bossak_coefficients(alpha=float(bossak_alpha), dt=float(dt))
    qorder = int(quadrature_order) if quadrature_order is not None else int(getattr(state, "quadrature_order", 2))
    coeffs = _dvms_state_coefficients(state)
    forms = build_fluid_dvms_local_forms(
        du=du,
        dp=dp,
        v=v,
        q=q,
        u_k=u_k,
        u_prev=u_k if u_prev is None else u_prev,
        a_prev=u_k if a_prev is None else a_prev,
        a_curr=a_curr,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        d_prev2=d_prev2,
        mesh_v=mesh_v,
        mesh_v_prev=mesh_v_prev,
        mesh_a_prev=mesh_a_prev,
        dx_measure=dx(metadata={"q": qorder}),
        rho=Constant(float(rho_f)),
        mu=Constant(float(mu_f)),
        dt=Constant(max(float(dt), 1.0e-14)),
        h=_kratos_dvms_element_size_coefficient(mesh) if h_coefficient is None else h_coefficient,
        bossak_ma0=Constant(float(bossak["ma0"])),
        bossak_ma2=Constant(float(bossak["ma2"])),
        bossak_alpha=Constant(float(bossak["alpha"])),
        predicted_subscale=coeffs["predicted_subscale"],
        old_subscale=coeffs["old_subscale"],
        momentum_projection=coeffs["momentum_projection"],
        mass_projection=coeffs["mass_projection"],
        old_mass_residual=coeffs["old_mass_residual"],
        body_force=None if body_force is None else np.asarray(body_force, dtype=float).reshape(2),
        use_oss=bool(use_oss),
    )
    if mode == "velocity":
        return Equation(forms.add_velocity_jacobian, forms.add_velocity_residual)
    if mode == "system":
        return Equation(forms.system_jacobian, forms.system_residual)
    if mode == "system_condensed":
        return Equation(forms.system_condensed_jacobian, forms.system_condensed_residual)
    if mode == "mass_lhs":
        return forms.mass_lhs
    return forms.mass_stabilization


def _build_fluid_dvms_kratos_residualized_system_forms(
    *,
    mesh: Mesh,
    dh: DofHandler,
    u_k: VectorFunction,
    u_prev: VectorFunction | None,
    a_prev: VectorFunction | None,
    a_curr: VectorFunction | None,
    p_k: Function,
    d_mesh: VectorFunction,
    d_prev: VectorFunction,
    d_prev2: VectorFunction | None,
    mesh_v: VectorFunction | None,
    mesh_v_prev: VectorFunction | None,
    mesh_a_prev: VectorFunction | None,
    state: FluidDVMSState | None,
    rho_f: float,
    mu_f: float,
    dt: float,
    bossak_alpha: float,
    quadrature_order: int | None,
    body_force: np.ndarray | None,
    use_oss: bool,
    h_coefficient: ElementWiseConstant | None = None,
) -> _KratosResidualizedSystemForms:
    if u_prev is None or a_prev is None:
        raise ValueError("Kratos residualized DVMS system requires u_prev and a_prev.")

    du, dp, v, q = _local_trial_test_functions(dh)
    bossak = _bossak_coefficients(alpha=float(bossak_alpha), dt=float(dt))
    qorder = int(quadrature_order) if quadrature_order is not None else int(getattr(state, "quadrature_order", 2))
    coeffs = _dvms_state_coefficients(state)
    split = build_fluid_dvms_kratos_split_forms(
        du=du,
        dp=dp,
        v=v,
        q=q,
        u_k=u_k,
        u_prev=u_prev,
        a_prev=a_prev,
        a_curr=a_curr,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        d_prev2=d_prev2,
        mesh_v=mesh_v,
        mesh_v_prev=mesh_v_prev,
        mesh_a_prev=mesh_a_prev,
        dx_measure=dx(metadata={"q": qorder}),
        rho=Constant(float(rho_f)),
        mu=Constant(float(mu_f)),
        dt=Constant(max(float(dt), 1.0e-14)),
        h=_kratos_dvms_element_size_coefficient(mesh) if h_coefficient is None else h_coefficient,
        bossak_ma0=Constant(float(bossak["ma0"])),
        bossak_ma2=Constant(float(bossak["ma2"])),
        bossak_alpha=Constant(float(bossak["alpha"])),
        predicted_subscale=coeffs["predicted_subscale"],
        old_subscale=coeffs["old_subscale"],
        momentum_projection=coeffs["momentum_projection"],
        mass_projection=coeffs["mass_projection"],
        old_mass_residual=coeffs["old_mass_residual"],
        body_force=None if body_force is None else np.asarray(body_force, dtype=float).reshape(2),
        use_oss=bool(use_oss),
    )
    nonviscous_lhs = _sum_forms(
        form for name, form in split.lhs_terms.items() if str(name) != "viscous"
    )
    source_rhs = _sum_forms(
        form
        for name, form in split.rhs_terms.items()
        if str(name) != "viscous" and not str(name).startswith("minus_")
    )
    viscous_lhs = split.lhs_terms["viscous"]
    viscous_rhs = split.rhs_terms["viscous"]
    mass_matrix = split.mass_terms["mass_lhs"] + split.mass_terms["mass_stabilization"]
    return _KratosResidualizedSystemForms(
        nonviscous_velocity=Equation(nonviscous_lhs, source_rhs),
        viscous_velocity=Equation(viscous_lhs, viscous_rhs),
        mass_matrix=mass_matrix,
    )


def assemble_fluid_dvms_local_contribution_batch(
    *,
    mesh: Mesh,
    dh: DofHandler,
    u_k: VectorFunction,
    u_prev: VectorFunction | None = None,
    a_prev: VectorFunction | None = None,
    a_curr: VectorFunction | None = None,
    p_k: Function,
    d_mesh: VectorFunction,
    d_prev: VectorFunction,
    d_prev2: VectorFunction | None = None,
    mesh_v: VectorFunction | None = None,
    mesh_v_prev: VectorFunction | None = None,
    mesh_a_prev: VectorFunction | None = None,
    state: FluidDVMSState | None,
    rho_f: float,
    mu_f: float,
    dt: float,
    bossak_alpha: float = -0.3,
    element_ids: np.ndarray | None = None,
    quadrature_order: int | None = None,
    body_force: np.ndarray | None = None,
    use_oss: bool = False,
    contribution_mode: str = "velocity",
    backend: str = "python",
):
    compiler = FormCompiler(
        dh,
        quadrature_order=None if quadrature_order is None else int(quadrature_order),
        backend=_normalize_backend_name(backend),
    )
    form_or_equation = _build_fluid_dvms_form_or_equation(
        mesh=mesh,
        dh=dh,
        u_k=u_k,
        u_prev=u_prev,
        a_prev=a_prev,
        a_curr=a_curr,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        d_prev2=d_prev2,
        mesh_v=mesh_v,
        mesh_v_prev=mesh_v_prev,
        mesh_a_prev=mesh_a_prev,
        state=state,
        rho_f=float(rho_f),
        mu_f=float(mu_f),
        dt=float(dt),
        bossak_alpha=float(bossak_alpha),
        quadrature_order=quadrature_order,
        body_force=body_force,
        use_oss=bool(use_oss),
        contribution_mode=contribution_mode,
        h_coefficient=_kratos_dvms_current_element_size_coefficient(mesh, dh, d_mesh),
    )
    return compiler.assemble_volume_local_contributions(form_or_equation, element_ids=element_ids)


def _single_element_batch_result(batch):
    if int(batch.element_ids.shape[0]) != 1:
        raise ValueError("Expected a single-element local contribution batch.")
    lhs = None if batch.K_elem is None else np.asarray(batch.K_elem[0], dtype=float)
    rhs = None if batch.F_elem is None else np.asarray(batch.F_elem[0], dtype=float)
    gdofs = np.asarray(batch.gdofs_map[0], dtype=int)
    return lhs, rhs, gdofs


def assemble_dvms_add_velocity_system_local(
    *,
    mesh: Mesh,
    dh: DofHandler,
    eid: int,
    u_k: VectorFunction,
    p_k: Function,
    d_mesh: VectorFunction,
    d_prev: VectorFunction,
    state: FluidDVMSState | None,
    rho_f: float,
    mu_f: float,
    dt: float,
    quadrature_order: int | None = None,
    body_force: np.ndarray | None = None,
    backend: str = "python",
):
    batch = assemble_fluid_dvms_local_contribution_batch(
        mesh=mesh,
        dh=dh,
        u_k=u_k,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        state=state,
        rho_f=rho_f,
        mu_f=mu_f,
        dt=dt,
        element_ids=np.asarray([int(eid)], dtype=int),
        quadrature_order=quadrature_order,
        body_force=body_force,
        contribution_mode="velocity",
        backend=backend,
    )
    lhs_batch, rhs_batch, gdofs_batch = _compress_batch_to_fluid_block(dh, batch)
    batch = LocalAssemblyResult(
        K_elem=lhs_batch,
        F_elem=rhs_batch,
        element_ids=np.asarray(batch.element_ids, dtype=int),
        gdofs_map=gdofs_batch,
    )
    lhs, rhs, gdofs = _single_element_batch_result(batch)
    return lhs, rhs, gdofs


def assemble_dvms_add_mass_lhs_local(
    *,
    mesh: Mesh,
    dh: DofHandler,
    eid: int,
    d_mesh: VectorFunction,
    rho_f: float,
    quadrature_order: int | None = None,
    backend: str = "python",
) -> tuple[np.ndarray, np.ndarray]:
    zero_u = VectorFunction("u_mass_local", ["ux", "uy"], dof_handler=dh)
    zero_p = Function("p_mass_local", "p", dof_handler=dh)
    zero_d_prev = VectorFunction("d_prev_mass_local", ["mx", "my"], dof_handler=dh)
    batch = assemble_fluid_dvms_local_contribution_batch(
        mesh=mesh,
        dh=dh,
        u_k=zero_u,
        p_k=zero_p,
        d_mesh=d_mesh,
        d_prev=zero_d_prev,
        state=None,
        rho_f=rho_f,
        mu_f=0.0,
        dt=1.0,
        element_ids=np.asarray([int(eid)], dtype=int),
        quadrature_order=quadrature_order,
        contribution_mode="mass_lhs",
        backend=backend,
    )
    lhs_batch, _, gdofs_batch = _compress_batch_to_fluid_block(dh, batch)
    batch = LocalAssemblyResult(
        K_elem=lhs_batch,
        F_elem=None,
        element_ids=np.asarray(batch.element_ids, dtype=int),
        gdofs_map=gdofs_batch,
    )
    lhs, _, gdofs = _single_element_batch_result(batch)
    return lhs, gdofs


def assemble_dvms_add_mass_stabilization_local(
    *,
    mesh: Mesh,
    dh: DofHandler,
    eid: int,
    u_k: VectorFunction,
    p_k: Function,
    d_mesh: VectorFunction,
    d_prev: VectorFunction,
    state: FluidDVMSState | None,
    rho_f: float,
    mu_f: float,
    dt: float,
    quadrature_order: int | None = None,
    backend: str = "python",
) -> tuple[np.ndarray, np.ndarray]:
    zero_u_prev = VectorFunction("u_prev_mass_stab_local", ["ux", "uy"], dof_handler=dh)
    zero_a_prev = VectorFunction("a_prev_mass_stab_local", ["ux", "uy"], dof_handler=dh)
    batch = assemble_fluid_dvms_local_contribution_batch(
        mesh=mesh,
        dh=dh,
        u_k=u_k,
        u_prev=zero_u_prev,
        a_prev=zero_a_prev,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        state=state,
        rho_f=rho_f,
        mu_f=mu_f,
        dt=dt,
        element_ids=np.asarray([int(eid)], dtype=int),
        quadrature_order=quadrature_order,
        contribution_mode="mass_stabilization",
        backend=backend,
    )
    lhs_batch, _, gdofs_batch = _compress_batch_to_fluid_block(dh, batch)
    batch = LocalAssemblyResult(
        K_elem=lhs_batch,
        F_elem=None,
        element_ids=np.asarray(batch.element_ids, dtype=int),
        gdofs_map=gdofs_batch,
    )
    lhs, _, gdofs = _single_element_batch_result(batch)
    return lhs, gdofs


def assemble_dvms_calculate_local_velocity_contribution(
    *,
    mesh: Mesh,
    dh: DofHandler,
    eid: int,
    u_k: VectorFunction,
    p_k: Function,
    d_mesh: VectorFunction,
    d_prev: VectorFunction,
    state: FluidDVMSState | None,
    rho_f: float,
    mu_f: float,
    dt: float,
    bossak_alpha: float = -0.3,
    quadrature_order: int | None = None,
    body_force: np.ndarray | None = None,
    use_oss: bool = False,
    backend: str = "python",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del bossak_alpha, use_oss
    return assemble_dvms_add_velocity_system_local(
        mesh=mesh,
        dh=dh,
        eid=eid,
        u_k=u_k,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        state=state,
        rho_f=rho_f,
        mu_f=mu_f,
        dt=dt,
        quadrature_order=quadrature_order,
        body_force=body_force,
        backend=backend,
    )


def assemble_dvms_calculate_local_system(
    *,
    mesh: Mesh,
    dh: DofHandler,
    eid: int,
    u_k: VectorFunction,
    u_prev: VectorFunction,
    a_prev: VectorFunction,
    p_k: Function,
    d_mesh: VectorFunction,
    d_prev: VectorFunction,
    state: FluidDVMSState | None,
    rho_f: float,
    mu_f: float,
    dt: float,
    bossak_alpha: float = -0.3,
    quadrature_order: int | None = None,
    body_force: np.ndarray | None = None,
    use_oss: bool = False,
    backend: str = "python",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch = assemble_fluid_dvms_local_contribution_batch(
        mesh=mesh,
        dh=dh,
        u_k=u_k,
        u_prev=u_prev,
        a_prev=a_prev,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        state=state,
        rho_f=rho_f,
        mu_f=mu_f,
        dt=dt,
        bossak_alpha=bossak_alpha,
        element_ids=np.asarray([int(eid)], dtype=int),
        quadrature_order=quadrature_order,
        body_force=body_force,
        use_oss=use_oss,
        contribution_mode="system",
        backend=backend,
    )
    lhs_batch, rhs_batch, gdofs_batch = _compress_batch_to_fluid_block(dh, batch)
    batch = LocalAssemblyResult(
        K_elem=lhs_batch,
        F_elem=rhs_batch,
        element_ids=np.asarray(batch.element_ids, dtype=int),
        gdofs_map=gdofs_batch,
    )
    lhs, rhs, gdofs = _single_element_batch_result(batch)
    return lhs, rhs, gdofs


class FluidDVMSLocalVelocityContributionOperator(LocalAssemblyOperator):
    """
    Symbolic DVMS local-block operator backed by the repo-level local batch API.

    This keeps the example-side policy in `examples/NIRB` while routing the
    actual element matrix/vector evaluation through the shared UFL compiler for
    python, jit, and cpp backends.
    """

    def __init__(
        self,
        *,
        mesh: Mesh,
        dh: DofHandler,
        u_k: VectorFunction,
        u_prev: VectorFunction | None = None,
        a_prev: VectorFunction | None = None,
        a_curr: VectorFunction | None = None,
        p_k: Function,
        d_mesh: VectorFunction,
        d_prev: VectorFunction,
        d_prev2: VectorFunction | None = None,
        mesh_v: VectorFunction | None = None,
        mesh_v_prev: VectorFunction | None = None,
        mesh_a_prev: VectorFunction | None = None,
        state: FluidDVMSState | None,
        rho_f: float,
        mu_f: float,
        dt: float,
        bossak_alpha: float = -0.3,
        element_ids: np.ndarray | None = None,
        quadrature_order: int | None = None,
        body_force: np.ndarray | None = None,
        use_oss: bool = False,
        apply_dirichlet_lift: bool = False,
        contribution_mode: str = "velocity",
        residualization: str = "kratos",
    ) -> None:
        self.mesh = mesh
        self.dh = dh
        self.u_k = u_k
        self.u_prev = u_prev
        self.a_prev = a_prev
        self.a_curr = a_curr
        self.p_k = p_k
        self.d_mesh = d_mesh
        self.d_prev = d_prev
        self.d_prev2 = d_prev2
        self.mesh_v = mesh_v
        self.mesh_v_prev = mesh_v_prev
        self.mesh_a_prev = mesh_a_prev
        self.state = state
        self.rho_f = float(rho_f)
        self.mu_f = float(mu_f)
        self.dt = float(dt)
        self.bossak_alpha = float(bossak_alpha)
        self.element_ids = (
            np.arange(int(mesh.n_elements), dtype=int)
            if element_ids is None
            else np.asarray(element_ids, dtype=int).reshape(-1)
        )
        self.quadrature_order = None if quadrature_order is None else int(quadrature_order)
        self.body_force = None if body_force is None else np.asarray(body_force, dtype=float).reshape(2)
        self.use_oss = bool(use_oss)
        self.apply_dirichlet_lift = bool(apply_dirichlet_lift)
        self.contribution_mode = _normalize_contribution_mode(contribution_mode)
        residualization_name = str(residualization or "symbolic").strip().lower()
        if residualization_name not in {"symbolic", "kratos"}:
            raise ValueError(f"Unsupported FluidDVMS residualization mode {residualization!r}.")
        self.residualization = residualization_name
        self.h_coefficient = _kratos_dvms_current_element_size_coefficient(self.mesh, self.dh, self.d_mesh)
        self._compiler_cache: dict[str, FormCompiler] = {}
        self._form_or_equation_cache = None
        self._kratos_residualized_forms_cache: _KratosResidualizedSystemForms | None = None
        self._fluid_gdofs_map = _local_grouped_gdofs_batch(self.dh, self.element_ids)
        self._compiled_layout_cache_map: np.ndarray | None = None
        self._fluid_block_positions_cache: np.ndarray | None = None

    def _compiler(self, backend: str) -> FormCompiler:
        key = _normalize_backend_name(backend)
        cached = self._compiler_cache.get(key)
        if cached is not None:
            return cached
        compiler = FormCompiler(
            self.dh,
            quadrature_order=self.quadrature_order,
            backend=key,
        )
        self._compiler_cache[key] = compiler
        return compiler

    def _form_or_equation(self):
        cached = self._form_or_equation_cache
        if cached is not None:
            return cached
        cached = _build_fluid_dvms_form_or_equation(
            mesh=self.mesh,
            dh=self.dh,
            u_k=self.u_k,
            u_prev=self.u_prev,
            a_prev=self.a_prev,
            a_curr=self.a_curr,
            p_k=self.p_k,
            d_mesh=self.d_mesh,
            d_prev=self.d_prev,
            d_prev2=self.d_prev2,
            mesh_v=self.mesh_v,
            mesh_v_prev=self.mesh_v_prev,
            mesh_a_prev=self.mesh_a_prev,
            state=self.state,
            rho_f=self.rho_f,
            mu_f=self.mu_f,
            dt=self.dt,
            bossak_alpha=self.bossak_alpha,
            quadrature_order=self.quadrature_order,
            body_force=self.body_force,
            use_oss=self.use_oss,
            contribution_mode=self.contribution_mode,
            h_coefficient=self.h_coefficient,
        )
        self._form_or_equation_cache = cached
        return cached

    def _kratos_residualized_forms(self) -> _KratosResidualizedSystemForms:
        cached = self._kratos_residualized_forms_cache
        if cached is not None:
            return cached
        cached = _build_fluid_dvms_kratos_residualized_system_forms(
            mesh=self.mesh,
            dh=self.dh,
            u_k=self.u_k,
            u_prev=self.u_prev,
            a_prev=self.a_prev,
            a_curr=self.a_curr,
            p_k=self.p_k,
            d_mesh=self.d_mesh,
            d_prev=self.d_prev,
            d_prev2=self.d_prev2,
            mesh_v=self.mesh_v,
            mesh_v_prev=self.mesh_v_prev,
            mesh_a_prev=self.mesh_a_prev,
            state=self.state,
            rho_f=self.rho_f,
            mu_f=self.mu_f,
            dt=self.dt,
            bossak_alpha=self.bossak_alpha,
            quadrature_order=self.quadrature_order,
            body_force=self.body_force,
            use_oss=self.use_oss,
            h_coefficient=self.h_coefficient,
        )
        self._kratos_residualized_forms_cache = cached
        return cached

    def _refresh_element_size_coefficient(self) -> None:
        eids = np.asarray(self.element_ids, dtype=int).reshape(-1)
        if eids.size == int(self.mesh.n_elements):
            self.h_coefficient.values[...] = _kratos_dvms_current_element_size_array(
                self.mesh,
                self.dh,
                self.d_mesh,
            )
            return
        self.h_coefficient.values[eids] = _kratos_dvms_current_element_size_array(
            self.mesh,
            self.dh,
            self.d_mesh,
            element_ids=eids,
        )

    def build_local_workset(self, *, solver, coeffs, need_matrix: bool):
        del coeffs
        self._refresh_element_size_coefficient()
        payload: dict[str, np.ndarray] = {}
        if self.apply_dirichlet_lift:
            bcs_apply = getattr(solver, "_current_bcs", None)
            if bcs_apply is None:
                bcs_inhom = getattr(solver, "bcs", None)
                bcs_homog = getattr(solver, "bcs_homog", None)
                # Local lifting must use the actual inhomogeneous Dirichlet
                # data. Falling back to bcs_homog silently zeroes the lifted
                # residual and breaks exact reaction parity.
                bcs_apply = bcs_inhom if bcs_inhom else bcs_homog
            bc_map = self.dh.get_dirichlet_data(bcs_apply) or {}
            bc_values_full = np.zeros(int(self.dh.total_dofs), dtype=float)
            if bc_map:
                bc_ids = np.fromiter((int(gdof) for gdof in bc_map.keys()), dtype=int)
                bc_vals = np.fromiter((float(val) for val in bc_map.values()), dtype=float)
                bc_values_full[bc_ids] = bc_vals
            payload["bc_values_full"] = bc_values_full
        return LocalAssemblyWorkset(
            solver=solver,
            coeffs=None,
            need_matrix=bool(need_matrix),
            backend=str(getattr(solver, "backend", "python")),
            element_ids=self.element_ids,
            gdofs_map=self._fluid_gdofs_map,
            payload=payload,
        )

    def _fluid_block_positions(self, batch_gdofs_map: np.ndarray) -> np.ndarray:
        batch_map = np.asarray(batch_gdofs_map, dtype=int)
        cached_map = self._compiled_layout_cache_map
        cached_pos = self._fluid_block_positions_cache
        if (
            cached_map is not None
            and cached_pos is not None
            and cached_map.shape == batch_map.shape
            and np.array_equal(cached_map, batch_map)
        ):
            return cached_pos
        positions = _fluid_local_layout_positions(batch_map, self._fluid_gdofs_map)
        self._compiled_layout_cache_map = batch_map.copy()
        self._fluid_block_positions_cache = positions
        return positions

    def _compress_compiler_batch(self, batch):
        batch_gdofs = np.asarray(batch.gdofs_map, dtype=int)
        positions = self._fluid_block_positions(batch_gdofs)
        return _compress_batch_to_fluid_block(
            self.dh,
            LocalAssemblyResult(
                K_elem=None if batch.K_elem is None else np.asarray(batch.K_elem, dtype=float),
                F_elem=None if batch.F_elem is None else np.asarray(batch.F_elem, dtype=float),
                element_ids=np.asarray(batch.element_ids, dtype=int),
                gdofs_map=batch_gdofs,
            ),
            fluid_gdofs_map=self._fluid_gdofs_map,
            positions=positions,
        )

    def _assemble_kratos_residualized_system_local(self, workset: LocalAssemblyWorkset):
        compiler = self._compiler(workset.backend)
        forms = self._kratos_residualized_forms()

        nonvisc_batch = compiler.assemble_volume_local_contributions(
            forms.nonviscous_velocity,
            element_ids=workset.element_ids,
            need_matrix=True,
            need_vector=True,
        )
        K_nonvisc, F_source, gdofs_map = self._compress_compiler_batch(nonvisc_batch)
        if K_nonvisc is None or F_source is None:
            raise RuntimeError("Kratos residualized DVMS nonviscous assembly did not return both K and RHS.")

        visc_batch = compiler.assemble_volume_local_contributions(
            forms.viscous_velocity,
            element_ids=workset.element_ids,
            need_matrix=True,
            need_vector=True,
        )
        K_visc, F_visc, _ = self._compress_compiler_batch(visc_batch)
        if K_visc is None or F_visc is None:
            raise RuntimeError("Kratos residualized DVMS viscous assembly did not return both K and RHS.")

        mass_batch = compiler.assemble_volume_local_contributions(
            forms.mass_matrix,
            element_ids=workset.element_ids,
            need_matrix=True,
            need_vector=False,
        )
        M_elem, _, _ = self._compress_compiler_batch(mass_batch)
        if M_elem is None:
            raise RuntimeError("Kratos residualized DVMS mass assembly did not return K.")

        x_full = _fluid_values_full(self.dh, u=self.u_k, p=self.p_k)
        a_full = _fluid_bossak_acceleration_values_full(
            self.dh,
            u_k=self.u_k,
            u_prev=self.u_prev,
            a_prev=self.a_prev,
            a_curr=self.a_curr,
            bossak_alpha=self.bossak_alpha,
            dt=self.dt,
        )
        local_values = np.asarray(x_full[np.asarray(gdofs_map, dtype=int)], dtype=float)
        local_acceleration = np.asarray(a_full[np.asarray(gdofs_map, dtype=int)], dtype=float)

        bossak = _bossak_coefficients(alpha=self.bossak_alpha, dt=self.dt)
        K_elem = np.asarray(K_nonvisc, dtype=float) + np.asarray(K_visc, dtype=float) + float(bossak["mam"]) * np.asarray(M_elem, dtype=float)
        F_elem = (
            np.asarray(F_source, dtype=float)
            + np.asarray(F_visc, dtype=float)
            - np.einsum("eij,ej->ei", np.asarray(K_nonvisc, dtype=float), local_values, optimize=True)
            - np.einsum("eij,ej->ei", np.asarray(M_elem, dtype=float), local_acceleration, optimize=True)
        )
        return K_elem, F_elem, gdofs_map

    def assemble_local(self, workset: LocalAssemblyWorkset):
        if self.contribution_mode == "system" and self.residualization == "kratos":
            K_elem, F_elem, gdofs_map = self._assemble_kratos_residualized_system_local(workset)
            batch_element_ids = np.asarray(workset.element_ids, dtype=int)
        else:
            compiler = self._compiler(workset.backend)
            batch = compiler.assemble_volume_local_contributions(
                self._form_or_equation(),
                element_ids=workset.element_ids,
            )
            K_elem, F_elem, gdofs_map = self._compress_compiler_batch(batch)
            batch_element_ids = np.asarray(batch.element_ids, dtype=int)
        if self.apply_dirichlet_lift and K_elem is not None:
            bc_values_full = workset.payload.get("bc_values_full")
            if bc_values_full is not None:
                local_bc = np.asarray(
                    bc_values_full[np.asarray(gdofs_map, dtype=int)],
                    dtype=float,
                )
                lifted = np.einsum(
                    "eij,ej->ei",
                    np.asarray(K_elem, dtype=float),
                    local_bc,
                    optimize=True,
                )
                if F_elem is None:
                    F_elem = -lifted
                else:
                    F_elem = np.asarray(F_elem, dtype=float) - lifted
        if F_elem is not None:
            # The symbolic DVMS local block mirrors Kratos' native local
            # system convention K * delta = RHS. The Newton solver, however,
            # expects the nonlinear residual F(u) and solves K * delta = -F(u).
            # Therefore the solver-facing runtime operator must inject -RHS.
            F_elem = -np.asarray(F_elem, dtype=float)
        return LocalAssemblyResult(
            K_elem=None if (K_elem is None or not workset.need_matrix) else K_elem,
            F_elem=F_elem,
            element_ids=batch_element_ids,
            gdofs_map=gdofs_map,
        )


FluidDVMSAddVelocityLocalOperator = FluidDVMSLocalVelocityContributionOperator


class FluidDVMSCondensedLocalSystemOperator(FluidDVMSLocalVelocityContributionOperator):
    """
    Backward-compatible exact local DVMS system operator.

    The class name is historical. For Kratos parity the assembled monolithic
    fluid system is the velocity contribution plus the Bossak mass blocks; the
    extra symbolic hidden-state Schur correction is not part of the Kratos
    local operator and introduces a measurable interface-row mismatch.

    This wrapper therefore keeps only the predictor refresh behavior and, by
    default, assembles the plain Kratos-matched ``system`` block.
    """

    def __init__(
        self,
        *,
        dynamic_tau: float = 1.0,
        refresh_predicted_subscale: bool = True,
        **kwargs,
    ) -> None:
        kwargs = dict(kwargs)
        kwargs.setdefault("contribution_mode", "system")
        super().__init__(**kwargs)
        self.dynamic_tau = float(dynamic_tau)
        self.refresh_predicted_subscale = bool(refresh_predicted_subscale)

    def assemble_local(self, workset: LocalAssemblyWorkset):
        if self.refresh_predicted_subscale and bool(workset.need_matrix):
            if self.state is None or self.u_prev is None or self.a_prev is None:
                raise ValueError(
                    "FluidDVMSCondensedLocalSystemOperator requires state, u_prev, and a_prev."
                )
            _update_fluid_dvms_predicted_subscale(
                state=self.state,
                dh=self.dh,
                mesh=self.mesh,
                u_k=self.u_k,
                u_prev=self.u_prev,
                a_prev=self.a_prev,
                a_curr=self.a_curr,
                p_k=self.p_k,
                d_mesh=self.d_mesh,
                d_prev=self.d_prev,
                d_prev2=self.d_prev2,
                mesh_v=self.mesh_v,
                mesh_v_prev=self.mesh_v_prev,
                mesh_a_prev=self.mesh_a_prev,
                rho_f=self.rho_f,
                mu_f=self.mu_f,
                dt=self.dt,
                bossak_alpha=self.bossak_alpha,
                dynamic_tau=self.dynamic_tau,
                backend=workset.backend,
                use_oss=bool(self.use_oss),
            )
        return super().assemble_local(workset)


# Backward-compatible aliases while the Example 2 scripts/tests migrate away from
# the old P1-specific names.
assemble_dvms_add_velocity_system_p1_tri = assemble_dvms_add_velocity_system_local
assemble_dvms_add_mass_lhs_p1_tri = assemble_dvms_add_mass_lhs_local
assemble_dvms_add_mass_stabilization_p1_tri = assemble_dvms_add_mass_stabilization_local
assemble_dvms_calculate_local_velocity_contribution_p1_tri = assemble_dvms_calculate_local_velocity_contribution
assemble_dvms_calculate_local_system_p1_tri = assemble_dvms_calculate_local_system


__all__ = [
    "FluidDVMSAddVelocityLocalOperator",
    "FluidDVMSCondensedLocalSystemOperator",
    "FluidDVMSLocalVelocityContributionOperator",
    "assemble_dvms_add_mass_lhs_local",
    "assemble_dvms_add_mass_lhs_p1_tri",
    "assemble_dvms_add_mass_stabilization_local",
    "assemble_dvms_add_mass_stabilization_p1_tri",
    "assemble_dvms_add_velocity_system_local",
    "assemble_dvms_add_velocity_system_p1_tri",
    "assemble_dvms_calculate_local_system",
    "assemble_dvms_calculate_local_system_p1_tri",
    "assemble_dvms_calculate_local_velocity_contribution",
    "assemble_dvms_calculate_local_velocity_contribution_p1_tri",
    "assemble_fluid_dvms_local_contribution_batch",
]
