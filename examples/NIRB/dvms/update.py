from __future__ import annotations

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import Constant, Function, Identity, VectorFunction, dot, grad, inv

from .helpers import (
    _bossak_coefficients,
    _field_values_on_global_dofs,
    _kratos_dvms_current_element_size_array,
    _kratos_dvms_current_element_size_coefficient,
)
from .state import FluidDVMSState
from .symbolics import build_fluid_dvms_old_mass_residual, build_fluid_dvms_predictor_symbolics


def _normalized_dvms_operator_backend(backend: str | None) -> str:
    name = str(backend or "python").strip().lower()
    if name in {"c++", "cpp"}:
        return "cpp"
    if name not in {"python", "jit", "cpp"}:
        return "python"
    return name


def _quadrature_expression_compiler(state: FluidDVMSState, dh: DofHandler, *, backend: str | None) -> FormCompiler:
    backend_name = _normalized_dvms_operator_backend(backend)
    cache = getattr(state, "_quadrature_expression_compilers", None)
    if not isinstance(cache, dict):
        cache = {}
        state._quadrature_expression_compilers = cache
    key = (int(id(dh)), str(backend_name))
    cached = cache.get(key)
    if isinstance(cached, FormCompiler):
        return cached
    compiler = FormCompiler(dh, quadrature_order=int(state.quadrature_order), backend=backend_name)
    cache[key] = compiler
    return compiler


def _all_element_ids(state: FluidDVMSState) -> np.ndarray:
    return np.arange(int(state.n_elements), dtype=int)


def _first_bad_element(values: np.ndarray) -> int | None:
    bad = np.argwhere(~np.isfinite(np.asarray(values, dtype=float)))
    if bad.size == 0:
        return None
    return int(bad[0, 0])


def _dvms_fast_p1_tri_cache(state: FluidDVMSState, dh: DofHandler, mesh: Mesh) -> dict[str, object] | None:
    cache = getattr(state, "_fast_p1_tri_cache", None)
    key = (int(id(dh)), int(id(mesh)))
    if isinstance(cache, dict) and key in cache:
        return cache[key]
    me = getattr(dh, "mixed_element", None)
    if me is None or str(getattr(mesh, "element_type", "")).strip().lower() != "tri":
        return None
    required_fields = ("ux", "uy", "p", "mx", "my")
    try:
        if any(int(np.asarray(dh.element_maps[fld]).shape[1]) != 3 for fld in required_fields):
            return None
    except Exception:
        return None
    ref_points = np.asarray(state.quadrature_layout.reference_points, dtype=float)
    if ref_points.ndim != 2 or ref_points.shape[1] != 2:
        return None
    try:
        geo = dh.precompute_geometric_factors(int(state.quadrature_order), reuse=True)
        J_inv = np.asarray(geo["J_inv"], dtype=float)
        detJ = np.asarray(geo["detJ"], dtype=float)
    except Exception:
        return None
    if J_inv.ndim != 4 or J_inv.shape[2:] != (2, 2):
        return None
    if detJ.ndim != 2:
        return None
    basis: dict[str, np.ndarray] = {}
    grad_ref: dict[str, np.ndarray] = {}
    try:
        for fld in required_fields:
            sl = me.slice(fld)
            basis[fld] = np.asarray(
                [np.asarray(me.basis(fld, float(xi), float(eta)), dtype=float)[sl] for xi, eta in ref_points],
                dtype=float,
            )
            grad_ref[fld] = np.asarray(
                np.asarray(me.grad_basis(fld, float(ref_points[0, 0]), float(ref_points[0, 1])), dtype=float)[sl, :],
                dtype=float,
            )
    except Exception:
        return None
    entry = {
        "basis": basis,
        "grad_ref": grad_ref,
        "J_inv": np.asarray(J_inv[:, 0, :, :], dtype=float),
        "detJ": np.asarray(detJ[:, 0], dtype=float),
        "element_maps": {fld: np.asarray(dh.element_maps[fld], dtype=int) for fld in required_fields},
    }
    if not isinstance(cache, dict):
        cache = {}
        state._fast_p1_tri_cache = cache
    cache[key] = entry
    return entry


def _scalar_locals(dh: DofHandler, field_name: str, values: np.ndarray, elem_map: np.ndarray) -> np.ndarray:
    scattered = _field_values_on_global_dofs(dh, field_name, values)
    return np.asarray(scattered[np.asarray(elem_map, dtype=int)], dtype=float)


def _grad_phi_phys(grad_ref: np.ndarray, J_inv: np.ndarray) -> np.ndarray:
    return np.einsum("lj,ejk->elk", np.asarray(grad_ref, dtype=float), np.asarray(J_inv, dtype=float), optimize=True)


def _solve_2x2_batched(A: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    detA = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    ok = np.isfinite(detA) & (np.abs(detA) > 1.0e-14) & np.all(np.isfinite(rhs), axis=-1)
    out = np.zeros_like(rhs, dtype=float)
    if np.any(ok):
        det_ok = detA[ok]
        rhs_ok = rhs[ok]
        A_ok = A[ok]
        out_ok = np.zeros_like(rhs_ok, dtype=float)
        out_ok[:, 0] = (A_ok[:, 1, 1] * rhs_ok[:, 0] - A_ok[:, 0, 1] * rhs_ok[:, 1]) / det_ok
        out_ok[:, 1] = (A_ok[:, 0, 0] * rhs_ok[:, 1] - A_ok[:, 1, 0] * rhs_ok[:, 0]) / det_ok
        out[ok, :] = out_ok
    return out, ok


def _update_fluid_dvms_state_from_previous_step(
    *,
    state: FluidDVMSState,
    dh: DofHandler,
    mesh: Mesh,
    u_prev: VectorFunction,
    d_prev: VectorFunction,
    d_geo: VectorFunction | None = None,
    backend: str | None = None,
    use_oss: bool = False,
) -> None:
    if int(state.sample_count) == 0:
        return
    fast = _dvms_fast_p1_tri_cache(state, dh, mesh)
    if fast is not None:
        n_elem = int(state.n_elements)
        n_q = int(state.n_qp_per_element)
        basis_u = np.asarray(fast["basis"]["ux"], dtype=float)
        geom_disp = d_prev if d_geo is None else d_geo
        ux_prev = _scalar_locals(dh, "ux", u_prev.components[0].nodal_values, fast["element_maps"]["ux"])
        uy_prev = _scalar_locals(dh, "uy", u_prev.components[1].nodal_values, fast["element_maps"]["uy"])
        mx_geo = _scalar_locals(dh, "mx", geom_disp.components[0].nodal_values, fast["element_maps"]["mx"])
        my_geo = _scalar_locals(dh, "my", geom_disp.components[1].nodal_values, fast["element_maps"]["my"])
        grad_phi_u = _grad_phi_phys(fast["grad_ref"]["ux"], fast["J_inv"])
        grad_phi_mx = _grad_phi_phys(fast["grad_ref"]["mx"], fast["J_inv"])
        grad_phi_my = _grad_phi_phys(fast["grad_ref"]["my"], fast["J_inv"])
        grad_ux = np.einsum("el,elk->ek", ux_prev, grad_phi_u, optimize=True)
        grad_uy = np.einsum("el,elk->ek", uy_prev, grad_phi_u, optimize=True)
        grad_mx = np.einsum("el,elk->ek", mx_geo, grad_phi_mx, optimize=True)
        grad_my = np.einsum("el,elk->ek", my_geo, grad_phi_my, optimize=True)
        F_geo = np.zeros((int(state.n_elements), 2, 2), dtype=float)
        F_geo[:, 0, 0] = 1.0 + grad_mx[:, 0]
        F_geo[:, 0, 1] = grad_mx[:, 1]
        F_geo[:, 1, 0] = grad_my[:, 0]
        F_geo[:, 1, 1] = 1.0 + grad_my[:, 1]
        detF = F_geo[:, 0, 0] * F_geo[:, 1, 1] - F_geo[:, 0, 1] * F_geo[:, 1, 0]
        bad = np.argwhere(~np.isfinite(detF) | (np.abs(detF) <= 1.0e-14))
        if bad.size:
            raise RuntimeError(f"Singular previous ALE deformation gradient on element {int(bad[0, 0])}.")
        cof = np.zeros_like(F_geo)
        cof[:, 0, 0] = F_geo[:, 1, 1]
        # Kratos' divergence uses the current-configuration DN_DX. In ALE
        # pull-back form this is cof(F):grad(u_ref) / det(F), with cof(F)
        # equal to J * F^{-T}. The off-diagonal entries are therefore the
        # cofactor matrix, not the adjugate used for F^{-1}.
        cof[:, 0, 1] = -F_geo[:, 1, 0]
        cof[:, 1, 0] = -F_geo[:, 0, 1]
        cof[:, 1, 1] = F_geo[:, 0, 0]
        grad_u = np.zeros_like(F_geo)
        grad_u[:, 0, :] = grad_ux
        grad_u[:, 1, :] = grad_uy
        old_mass = np.broadcast_to(
            (-np.einsum("eij,eij->e", cof, grad_u, optimize=True) / detF)[:, None],
            (n_elem, n_q),
        ).copy()
        prev_divproj_nodal = getattr(state, "_prev_nodal_div_projection", None)
        if bool(use_oss) and prev_divproj_nodal is not None:
            prev_divproj_nodal = np.asarray(prev_divproj_nodal, dtype=float).reshape(-1)
            ux_ids = np.asarray(fast["element_maps"]["ux"], dtype=int)
            if ux_ids.size and int(np.max(ux_ids)) < int(prev_divproj_nodal.shape[0]):
                old_divproj_local = np.asarray(prev_divproj_nodal[ux_ids], dtype=float)
                old_mass -= np.einsum("el,ql->eq", old_divproj_local, basis_u, optimize=True)
        state._reshape_scalar_quadrature(state.old_mass_residual)[...] = old_mass
        state.sync_coefficient("old_mass_residual")
        return

    compiler = _quadrature_expression_compiler(state, dh, backend=backend)
    expr = build_fluid_dvms_old_mass_residual(u_prev=u_prev, d_prev=d_prev, d_geo=d_geo)
    values = compiler.evaluate_volume_expressions_on_quadrature(
        {"old_mass_residual": expr},
        layout=state.quadrature_layout,
        element_ids=_all_element_ids(state),
    )["old_mass_residual"]

    bad = _first_bad_element(values)
    if bad is not None:
        raise RuntimeError(f"Singular previous ALE deformation gradient on element {bad}.")

    state._reshape_scalar_quadrature(state.old_mass_residual)[...] = np.asarray(values, dtype=float)
    state.sync_coefficients_from_samples()


def _clear_fluid_dvms_oss_projections(state: FluidDVMSState) -> None:
    if int(state.sample_count) == 0:
        return
    state.momentum_projection[:, :] = 0.0
    state.mass_projection[:] = 0.0
    state.sync_coefficient("momentum_projection")
    state.sync_coefficient("mass_projection")
    for name in (
        "_nodal_momentum_projection",
        "_nodal_div_projection",
        "_prev_nodal_div_projection",
    ):
        if hasattr(state, name):
            try:
                delattr(state, name)
            except Exception:
                setattr(state, name, None)


def _update_fluid_dvms_oss_projections(
    *,
    state: FluidDVMSState,
    dh: DofHandler,
    mesh: Mesh,
    u_k: VectorFunction,
    p_k: Function,
    d_mesh: VectorFunction,
    d_prev: VectorFunction,
    d_prev2: VectorFunction | None = None,
    mesh_v: VectorFunction | None = None,
    mesh_v_prev: VectorFunction | None = None,
    mesh_a_prev: VectorFunction | None = None,
    rho_f: float,
    dt: float,
    bossak_alpha: float,
    body_force: np.ndarray | None = None,
) -> None:
    if int(state.sample_count) == 0:
        return
    fast = _dvms_fast_p1_tri_cache(state, dh, mesh)
    if fast is None:
        raise NotImplementedError(
            "DVMS OSS projection update currently requires the fast P1-triangle fluid path."
        )

    n_elem = int(state.n_elements)
    n_q = int(state.n_qp_per_element)
    basis_u = np.asarray(fast["basis"]["ux"], dtype=float)
    q_weights = np.asarray(state.sample_ref_weights, dtype=float).reshape(n_elem, n_q)
    grad_phi_u = _grad_phi_phys(fast["grad_ref"]["ux"], fast["J_inv"])
    grad_phi_mx = _grad_phi_phys(fast["grad_ref"]["mx"], fast["J_inv"])
    grad_phi_my = _grad_phi_phys(fast["grad_ref"]["my"], fast["J_inv"])

    ux_vals = _scalar_locals(dh, "ux", u_k.components[0].nodal_values, fast["element_maps"]["ux"])
    uy_vals = _scalar_locals(dh, "uy", u_k.components[1].nodal_values, fast["element_maps"]["uy"])
    p_vals = _scalar_locals(dh, "p", p_k.nodal_values, fast["element_maps"]["p"])
    mx_vals = _scalar_locals(dh, "mx", d_mesh.components[0].nodal_values, fast["element_maps"]["mx"])
    my_vals = _scalar_locals(dh, "my", d_mesh.components[1].nodal_values, fast["element_maps"]["my"])
    mx_prev = _scalar_locals(dh, "mx", d_prev.components[0].nodal_values, fast["element_maps"]["mx"])
    my_prev = _scalar_locals(dh, "my", d_prev.components[1].nodal_values, fast["element_maps"]["my"])
    if d_prev2 is not None:
        mx_prev2 = _scalar_locals(dh, "mx", d_prev2.components[0].nodal_values, fast["element_maps"]["mx"])
        my_prev2 = _scalar_locals(dh, "my", d_prev2.components[1].nodal_values, fast["element_maps"]["my"])
    else:
        mx_prev2 = np.asarray(mx_prev, dtype=float)
        my_prev2 = np.asarray(my_prev, dtype=float)

    u_q = np.stack(
        [
            np.einsum("el,ql->eq", ux_vals, basis_u, optimize=True),
            np.einsum("el,ql->eq", uy_vals, basis_u, optimize=True),
        ],
        axis=2,
    )
    d_q = np.stack(
        [
            np.einsum("el,ql->eq", mx_vals, basis_u, optimize=True),
            np.einsum("el,ql->eq", my_vals, basis_u, optimize=True),
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
    d_prev2_q = np.stack(
        [
            np.einsum("el,ql->eq", mx_prev2, basis_u, optimize=True),
            np.einsum("el,ql->eq", my_prev2, basis_u, optimize=True),
        ],
        axis=2,
    )

    grad_ux = np.einsum("el,elk->ek", ux_vals, grad_phi_u, optimize=True)
    grad_uy = np.einsum("el,elk->ek", uy_vals, grad_phi_u, optimize=True)
    grad_p = np.einsum("el,elk->ek", p_vals, grad_phi_u, optimize=True)
    grad_mx = np.einsum("el,elk->ek", mx_vals, grad_phi_mx, optimize=True)
    grad_my = np.einsum("el,elk->ek", my_vals, grad_phi_my, optimize=True)

    F = np.zeros((n_elem, 2, 2), dtype=float)
    F[:, 0, 0] = 1.0 + grad_mx[:, 0]
    F[:, 0, 1] = grad_mx[:, 1]
    F[:, 1, 0] = grad_my[:, 0]
    F[:, 1, 1] = 1.0 + grad_my[:, 1]
    detF = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
    bad = np.argwhere(~np.isfinite(detF) | (np.abs(detF) <= 1.0e-14))
    if bad.size:
        raise RuntimeError(f"Singular ALE deformation gradient on element {int(bad[0, 0])}.")
    Finv = np.zeros_like(F)
    Finv[:, 0, 0] = F[:, 1, 1] / detF
    Finv[:, 0, 1] = -F[:, 0, 1] / detF
    Finv[:, 1, 0] = -F[:, 1, 0] / detF
    Finv[:, 1, 1] = F[:, 0, 0] / detF

    grad_basis_cur = np.einsum("eik,ekj->eij", grad_phi_u, Finv, optimize=True)
    grad_u = np.zeros((n_elem, 2, 2), dtype=float)
    grad_u[:, 0, :] = np.einsum("el,eij->ej", ux_vals, grad_basis_cur, optimize=True)
    grad_u[:, 1, :] = np.einsum("el,eij->ej", uy_vals, grad_basis_cur, optimize=True)
    grad_p_phys = np.einsum("el,eij->ej", p_vals, grad_basis_cur, optimize=True)

    if mesh_v is not None:
        mesh_vx = _scalar_locals(dh, "mx", mesh_v.components[0].nodal_values, fast["element_maps"]["mx"])
        mesh_vy = _scalar_locals(dh, "my", mesh_v.components[1].nodal_values, fast["element_maps"]["my"])
        w_q = np.stack(
            [
                np.einsum("el,ql->eq", mesh_vx, basis_u, optimize=True),
                np.einsum("el,ql->eq", mesh_vy, basis_u, optimize=True),
            ],
            axis=2,
        )
    elif mesh_v_prev is not None and mesh_a_prev is not None:
        bossak = _bossak_coefficients(alpha=float(bossak_alpha), dt=max(float(dt), 1.0e-14))
        beta = float(bossak["beta"])
        gamma = float(bossak["gamma"])
        mesh_v_prev_x = _scalar_locals(dh, "mx", mesh_v_prev.components[0].nodal_values, fast["element_maps"]["mx"])
        mesh_v_prev_y = _scalar_locals(dh, "my", mesh_v_prev.components[1].nodal_values, fast["element_maps"]["my"])
        mesh_a_prev_x = _scalar_locals(dh, "mx", mesh_a_prev.components[0].nodal_values, fast["element_maps"]["mx"])
        mesh_a_prev_y = _scalar_locals(dh, "my", mesh_a_prev.components[1].nodal_values, fast["element_maps"]["my"])
        mesh_v_prev_q = np.stack(
            [
                np.einsum("el,ql->eq", mesh_v_prev_x, basis_u, optimize=True),
                np.einsum("el,ql->eq", mesh_v_prev_y, basis_u, optimize=True),
            ],
            axis=2,
        )
        mesh_a_prev_q = np.stack(
            [
                np.einsum("el,ql->eq", mesh_a_prev_x, basis_u, optimize=True),
                np.einsum("el,ql->eq", mesh_a_prev_y, basis_u, optimize=True),
            ],
            axis=2,
        )
        dt_value = max(float(dt), 1.0e-14)
        a_mesh_q = (
            d_q
            - d_prev_q
            - dt_value * mesh_v_prev_q
            - (dt_value * dt_value) * (0.5 - beta) * mesh_a_prev_q
        ) / (beta * dt_value * dt_value)
        w_q = mesh_v_prev_q + dt_value * ((1.0 - gamma) * mesh_a_prev_q + gamma * a_mesh_q)
    else:
        dt_value = max(float(dt), 1.0e-14)
        w_q = (1.5 * d_q - 2.0 * d_prev_q + 0.5 * d_prev2_q) / dt_value

    predicted = np.asarray(state.predicted_subscale_velocity, dtype=float).reshape(n_elem, n_q, 2)
    conv_velocity = u_q - w_q + predicted
    momentum_res = (
        np.asarray(body_force if body_force is not None else np.zeros((2,), dtype=float), dtype=float).reshape(1, 1, 2)
        - np.einsum("eij,eqj->eqi", grad_u, conv_velocity, optimize=True)
    )
    momentum_res *= float(rho_f)
    momentum_res -= grad_p_phys[:, None, :]
    mass_res = -np.trace(grad_u, axis1=1, axis2=2)[:, None]

    weights = np.asarray(q_weights, dtype=float) * np.asarray(fast["detJ"], dtype=float)[:, None] * detF[:, None]
    weighted_phi = weights[:, :, None] * basis_u[None, :, :]

    ux_ids = np.asarray(fast["element_maps"]["ux"], dtype=int)
    ndof = int(np.max(ux_ids)) + 1 if ux_ids.size else 0
    nodal_area = np.zeros((ndof,), dtype=float)
    nodal_divproj = np.zeros((ndof,), dtype=float)
    nodal_advproj = np.zeros((ndof, 2), dtype=float)
    for a in range(weighted_phi.shape[2]):
        ids = np.asarray(ux_ids[:, a], dtype=int).reshape(-1)
        contrib_area = np.sum(weighted_phi[:, :, a], axis=1)
        contrib_div = np.sum(weighted_phi[:, :, a] * mass_res, axis=1)
        contrib_adv = np.sum(weighted_phi[:, :, a][:, :, None] * momentum_res, axis=1)
        np.add.at(nodal_area, ids, contrib_area)
        np.add.at(nodal_divproj, ids, contrib_div)
        np.add.at(nodal_advproj[:, 0], ids, contrib_adv[:, 0])
        np.add.at(nodal_advproj[:, 1], ids, contrib_adv[:, 1])

    active = np.abs(nodal_area) > 1.0e-30
    nodal_divproj[active] = nodal_divproj[active] / nodal_area[active]
    nodal_advproj[active, :] = nodal_advproj[active, :] / nodal_area[active, None]

    state._nodal_div_projection = np.asarray(nodal_divproj, dtype=float)
    state._nodal_momentum_projection = np.asarray(nodal_advproj, dtype=float)

    elem_divproj = np.asarray(nodal_divproj[ux_ids], dtype=float)
    elem_advproj = np.asarray(nodal_advproj[ux_ids], dtype=float)
    mass_q = np.einsum("ql,el->eq", basis_u, elem_divproj, optimize=True)
    momentum_q = np.einsum("ql,elc->eqc", basis_u, elem_advproj, optimize=True)
    state.mass_projection[:] = mass_q.reshape(int(state.sample_count))
    state.momentum_projection[:, :] = momentum_q.reshape(int(state.sample_count), 2)
    state.sync_coefficient("mass_projection")
    state.sync_coefficient("momentum_projection")


def _update_fluid_dvms_old_subscale_after_step(
    *,
    state: FluidDVMSState,
    dh: DofHandler,
    mesh: Mesh,
    u_curr: VectorFunction,
    a_curr: VectorFunction,
    p_curr: Function,
    d_curr: VectorFunction,
    mesh_v_curr: VectorFunction,
    rho_f: float,
    mu_f: float,
    dt: float,
    dynamic_tau: float,
    body_force: np.ndarray | None = None,
    backend: str | None = None,
    use_oss: bool = False,
) -> None:
    if int(state.sample_count) == 0:
        return
    fast = _dvms_fast_p1_tri_cache(state, dh, mesh)
    dt_value = max(float(dt), 1.0e-14)
    rho_value = float(rho_f)
    mu_value = float(mu_f)
    # Kratos' DVMS element stores DYNAMIC_TAU in the data container, but
    # d_vms.cpp hard-codes the dynamic subscale time term with coefficient 1.
    # Keep the exact local DVMS history update aligned with that implementation.
    del dynamic_tau
    dynamic_tau_value = 1.0
    body_force_value = (
        np.zeros((2,), dtype=float)
        if body_force is None
        else np.asarray(body_force, dtype=float).reshape(2)
    )
    if fast is not None:
        n_elem = int(state.n_elements)
        n_q = int(state.n_qp_per_element)
        basis_u = np.asarray(fast["basis"]["ux"], dtype=float)
        basis_p = np.asarray(fast["basis"]["p"], dtype=float)
        grad_phi_u = _grad_phi_phys(fast["grad_ref"]["ux"], fast["J_inv"])
        grad_phi_p = _grad_phi_phys(fast["grad_ref"]["p"], fast["J_inv"])
        grad_phi_mx = _grad_phi_phys(fast["grad_ref"]["mx"], fast["J_inv"])
        grad_phi_my = _grad_phi_phys(fast["grad_ref"]["my"], fast["J_inv"])

        ux_curr = _scalar_locals(dh, "ux", u_curr.components[0].nodal_values, fast["element_maps"]["ux"])
        uy_curr = _scalar_locals(dh, "uy", u_curr.components[1].nodal_values, fast["element_maps"]["uy"])
        ax_curr = _scalar_locals(dh, "ux", a_curr.components[0].nodal_values, fast["element_maps"]["ux"])
        ay_curr = _scalar_locals(dh, "uy", a_curr.components[1].nodal_values, fast["element_maps"]["uy"])
        p_vals = _scalar_locals(dh, "p", p_curr.nodal_values, fast["element_maps"]["p"])
        mx_curr = _scalar_locals(dh, "mx", d_curr.components[0].nodal_values, fast["element_maps"]["mx"])
        my_curr = _scalar_locals(dh, "my", d_curr.components[1].nodal_values, fast["element_maps"]["my"])
        mx_vel = _scalar_locals(dh, "mx", mesh_v_curr.components[0].nodal_values, fast["element_maps"]["mx"])
        my_vel = _scalar_locals(dh, "my", mesh_v_curr.components[1].nodal_values, fast["element_maps"]["my"])

        u_curr_q = np.stack(
            [
                np.einsum("el,ql->eq", ux_curr, basis_u, optimize=True),
                np.einsum("el,ql->eq", uy_curr, basis_u, optimize=True),
            ],
            axis=2,
        )
        a_curr_q = np.stack(
            [
                np.einsum("el,ql->eq", ax_curr, basis_u, optimize=True),
                np.einsum("el,ql->eq", ay_curr, basis_u, optimize=True),
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
        grad_u = np.zeros((n_elem, 2, 2), dtype=float)
        grad_u[:, 0, :] = np.einsum("el,elk->ek", ux_curr, grad_phi_u, optimize=True)
        grad_u[:, 1, :] = np.einsum("el,elk->ek", uy_curr, grad_phi_u, optimize=True)
        grad_d = np.zeros((n_elem, 2, 2), dtype=float)
        grad_d[:, 0, :] = np.einsum("el,elk->ek", mx_curr, grad_phi_mx, optimize=True)
        grad_d[:, 1, :] = np.einsum("el,elk->ek", my_curr, grad_phi_my, optimize=True)
        grad_p = np.einsum("el,elk->ek", p_vals, grad_phi_p, optimize=True)

        F = np.zeros((n_elem, 2, 2), dtype=float)
        F[:, 0, 0] = 1.0 + grad_d[:, 0, 0]
        F[:, 0, 1] = grad_d[:, 0, 1]
        F[:, 1, 0] = grad_d[:, 1, 0]
        F[:, 1, 1] = 1.0 + grad_d[:, 1, 1]
        detF = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
        bad = np.argwhere(~np.isfinite(detF) | (np.abs(detF) <= 1.0e-14))
        if bad.size:
            raise RuntimeError(f"Singular ALE deformation gradient on element {int(bad[0, 0])}.")
        Finv = np.zeros_like(F)
        Finv[:, 0, 0] = F[:, 1, 1] / detF
        Finv[:, 0, 1] = -F[:, 0, 1] / detF
        Finv[:, 1, 0] = -F[:, 1, 0] / detF
        Finv[:, 1, 1] = F[:, 0, 0] / detF
        grad_u_phys = np.einsum("eij,ejk->eik", grad_u, Finv, optimize=True)
        grad_p_phys = np.einsum("eji,ej->ei", Finv, grad_p, optimize=True)

        resolved_conv_velocity = u_curr_q - mesh_v_q
        predicted = np.asarray(state.predicted_subscale_velocity, dtype=float).reshape(n_elem, n_q, 2)
        old_subscale = np.asarray(state.old_subscale_velocity, dtype=float).reshape(n_elem, n_q, 2)
        if bool(use_oss):
            momentum_projection = np.asarray(state.momentum_projection, dtype=float).reshape(n_elem, n_q, 2)
        else:
            momentum_projection = np.zeros((n_elem, n_q, 2), dtype=float)
        conv_velocity = resolved_conv_velocity + predicted
        conv_speed = np.linalg.norm(conv_velocity, axis=2)
        h_e = _kratos_dvms_current_element_size_array(mesh, dh, d_curr).reshape(-1)
        if h_e.shape[0] != n_elem:
            raise RuntimeError(
                f"DVMS finalize element-size mismatch: expected {n_elem}, got {h_e.shape[0]}."
            )
        tau_one = dynamic_tau_value / (
            (8.0 * mu_value / np.maximum(h_e[:, None] * h_e[:, None], 1.0e-30))
            + rho_value * (1.0 / dt_value + 2.0 * conv_speed / np.maximum(h_e[:, None], 1.0e-30))
        )
        residual = rho_value * (
            body_force_value.reshape(1, 1, 2)
            - a_curr_q
            - np.einsum("eij,eqj->eqi", grad_u_phys, conv_velocity, optimize=True)
        )
        residual -= grad_p_phys[:, None, :]
        residual -= momentum_projection
        residual += (rho_value / dt_value) * old_subscale
        updated = tau_one[:, :, None] * residual
        if not np.all(np.isfinite(updated)):
            raise RuntimeError("DVMS finalize produced non-finite accepted subscale values.")
        state.old_subscale_velocity[:, :] = updated.reshape(int(state.sample_count), 2)
        state.sync_coefficient("old_subscale_velocity")
        return

    compiler = _quadrature_expression_compiler(state, dh, backend=backend)
    rho_expr = Constant(float(rho_value))
    mu_expr = Constant(float(mu_value))
    dt_expr = Constant(float(dt_value))
    dynamic_tau_expr = Constant(float(dynamic_tau_value))
    body_expr = Constant(np.asarray(body_force_value, dtype=float), dim=1)
    h_expr = _kratos_dvms_current_element_size_coefficient(mesh, dh, d_curr)
    F = Identity(2) + grad(d_curr)
    Finv = inv(F)
    grad_u_phys = dot(grad(u_curr), Finv)
    grad_p_phys = dot(Finv.T, grad(p_curr))
    conv_velocity = (u_curr - mesh_v_curr) + state.coefficient("predicted_subscale_velocity")
    conv_speed = dot(conv_velocity, conv_velocity) ** Constant(0.5)
    tau_one = dynamic_tau_expr / (
        Constant(8.0) * mu_expr / (h_expr * h_expr)
        + rho_expr * (Constant(1.0) / dt_expr + Constant(2.0) * conv_speed / h_expr)
    )
    momentum_projection_expr = (
        state.coefficient("momentum_projection")
        if bool(use_oss)
        else Constant(np.zeros((2,), dtype=float), dim=1)
    )
    accepted_old_subscale = tau_one * (
        rho_expr * (body_expr - a_curr - dot(grad_u_phys, conv_velocity))
        - grad_p_phys
        - momentum_projection_expr
        + (rho_expr / dt_expr) * state.coefficient("old_subscale_velocity")
    )
    values = compiler.evaluate_volume_expressions_on_quadrature(
        {"accepted_old_subscale": accepted_old_subscale},
        layout=state.quadrature_layout,
        element_ids=_all_element_ids(state),
    )["accepted_old_subscale"]
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("DVMS finalize produced non-finite accepted subscale values.")
    state.old_subscale_velocity[:, :] = values.reshape(int(state.sample_count), 2)
    state.sync_coefficient("old_subscale_velocity")


def _report_fluid_dvms_subscale_after_step(
    *,
    state: FluidDVMSState,
    dh: DofHandler,
    mesh: Mesh,
    u_curr: VectorFunction,
    a_curr: VectorFunction,
    p_curr: Function,
    d_curr: VectorFunction,
    mesh_v_curr: VectorFunction,
    rho_f: float,
    mu_f: float,
    dt: float,
    dynamic_tau: float,
    body_force: np.ndarray | None = None,
    backend: str | None = None,
    use_oss: bool = False,
) -> np.ndarray:
    """Evaluate Kratos' post-finalize SUBSCALE_VELOCITY report without mutating state."""
    saved_old_subscale = np.asarray(state.old_subscale_velocity, dtype=float).copy()
    _update_fluid_dvms_old_subscale_after_step(
        state=state,
        dh=dh,
        mesh=mesh,
        u_curr=u_curr,
        a_curr=a_curr,
        p_curr=p_curr,
        d_curr=d_curr,
        mesh_v_curr=mesh_v_curr,
        rho_f=float(rho_f),
        mu_f=float(mu_f),
        dt=float(dt),
        dynamic_tau=float(dynamic_tau),
        body_force=body_force,
        backend=backend,
        use_oss=bool(use_oss),
    )
    reported = np.asarray(state.old_subscale_velocity, dtype=float).copy()
    state.old_subscale_velocity[:, :] = saved_old_subscale
    state.sync_coefficient("old_subscale_velocity")
    return reported


def _update_fluid_dvms_predicted_subscale(
    *,
    state: FluidDVMSState,
    dh: DofHandler,
    mesh: Mesh,
    u_k: VectorFunction,
    u_prev: VectorFunction,
    a_prev: VectorFunction,
    a_curr: VectorFunction | None = None,
    p_k: Function,
    d_mesh: VectorFunction,
    d_prev: VectorFunction,
    d_prev2: VectorFunction | None = None,
    mesh_v: VectorFunction | None = None,
    mesh_v_prev: VectorFunction | None = None,
    mesh_a_prev: VectorFunction | None = None,
    rho_f: float,
    mu_f: float,
    dt: float,
    bossak_alpha: float,
    dynamic_tau: float,
    # Kratos DVMS hard-codes a maximum of 10 inner Newton iterations for the
    # hidden predicted subscale solve (d_vms.h / d_vms.cpp). Keep the local
    # default aligned with that exact stop rule.
    max_iterations: int = 10,
    rel_tol: float = 1.0e-14,
    abs_tol: float = 1.0e-14,
    backend: str | None = None,
    use_oss: bool = False,
) -> None:
    if int(state.sample_count) == 0:
        return
    fast = _dvms_fast_p1_tri_cache(state, dh, mesh)
    if fast is not None:
        dt_value = max(float(dt), 1.0e-14)
        rho_value = float(rho_f)
        mu_value = float(mu_f)
        # Kratos' DVMS predictor ignores DYNAMIC_TAU and uses 1 / dt directly.
        # The argument is retained for API compatibility with older callers.
        del dynamic_tau
        dynamic_tau_value = 1.0
        bossak = _bossak_coefficients(alpha=float(bossak_alpha), dt=dt_value)
        n_elem = int(state.n_elements)
        n_q = int(state.n_qp_per_element)
        basis_u = np.asarray(fast["basis"]["ux"], dtype=float)
        basis_p = np.asarray(fast["basis"]["p"], dtype=float)
        grad_phi_u = _grad_phi_phys(fast["grad_ref"]["ux"], fast["J_inv"])
        grad_phi_p = _grad_phi_phys(fast["grad_ref"]["p"], fast["J_inv"])
        grad_phi_mx = _grad_phi_phys(fast["grad_ref"]["mx"], fast["J_inv"])
        grad_phi_my = _grad_phi_phys(fast["grad_ref"]["my"], fast["J_inv"])

        ux_k = _scalar_locals(dh, "ux", u_k.components[0].nodal_values, fast["element_maps"]["ux"])
        uy_k = _scalar_locals(dh, "uy", u_k.components[1].nodal_values, fast["element_maps"]["uy"])
        ux_prev = _scalar_locals(dh, "ux", u_prev.components[0].nodal_values, fast["element_maps"]["ux"])
        uy_prev = _scalar_locals(dh, "uy", u_prev.components[1].nodal_values, fast["element_maps"]["uy"])
        ax_prev = _scalar_locals(dh, "ux", a_prev.components[0].nodal_values, fast["element_maps"]["ux"])
        ay_prev = _scalar_locals(dh, "uy", a_prev.components[1].nodal_values, fast["element_maps"]["uy"])
        if a_curr is not None:
            ax_curr = _scalar_locals(dh, "ux", a_curr.components[0].nodal_values, fast["element_maps"]["ux"])
            ay_curr = _scalar_locals(dh, "uy", a_curr.components[1].nodal_values, fast["element_maps"]["uy"])
        else:
            ax_curr = None
            ay_curr = None
        p_vals = _scalar_locals(dh, "p", p_k.nodal_values, fast["element_maps"]["p"])
        mx_k = _scalar_locals(dh, "mx", d_mesh.components[0].nodal_values, fast["element_maps"]["mx"])
        my_k = _scalar_locals(dh, "my", d_mesh.components[1].nodal_values, fast["element_maps"]["my"])
        mx_prev = _scalar_locals(dh, "mx", d_prev.components[0].nodal_values, fast["element_maps"]["mx"])
        my_prev = _scalar_locals(dh, "my", d_prev.components[1].nodal_values, fast["element_maps"]["my"])
        if mesh_v is not None:
            mx_vel = _scalar_locals(dh, "mx", mesh_v.components[0].nodal_values, fast["element_maps"]["mx"])
            my_vel = _scalar_locals(dh, "my", mesh_v.components[1].nodal_values, fast["element_maps"]["my"])
        else:
            mx_vel = np.zeros_like(mx_prev)
            my_vel = np.zeros_like(my_prev)
        if mesh_v_prev is not None and mesh_a_prev is not None:
            mx_vel_prev = _scalar_locals(dh, "mx", mesh_v_prev.components[0].nodal_values, fast["element_maps"]["mx"])
            my_vel_prev = _scalar_locals(dh, "my", mesh_v_prev.components[1].nodal_values, fast["element_maps"]["my"])
            mx_acc_prev = _scalar_locals(dh, "mx", mesh_a_prev.components[0].nodal_values, fast["element_maps"]["mx"])
            my_acc_prev = _scalar_locals(dh, "my", mesh_a_prev.components[1].nodal_values, fast["element_maps"]["my"])
        else:
            mx_vel_prev = np.zeros_like(mx_prev)
            my_vel_prev = np.zeros_like(my_prev)
            mx_acc_prev = np.zeros_like(mx_prev)
            my_acc_prev = np.zeros_like(my_prev)
        if d_prev2 is None:
            mx_prev2 = np.zeros_like(mx_prev)
            my_prev2 = np.zeros_like(my_prev)
        else:
            mx_prev2 = _scalar_locals(dh, "mx", d_prev2.components[0].nodal_values, fast["element_maps"]["mx"])
            my_prev2 = _scalar_locals(dh, "my", d_prev2.components[1].nodal_values, fast["element_maps"]["my"])

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
        a_curr_q = (
            np.stack(
                [
                    np.einsum("el,ql->eq", ax_curr, basis_u, optimize=True),
                    np.einsum("el,ql->eq", ay_curr, basis_u, optimize=True),
                ],
                axis=2,
            )
            if ax_curr is not None and ay_curr is not None
            else None
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
        grad_u = np.zeros((n_elem, 2, 2), dtype=float)
        grad_u[:, 0, :] = np.einsum("el,elk->ek", ux_k, grad_phi_u, optimize=True)
        grad_u[:, 1, :] = np.einsum("el,elk->ek", uy_k, grad_phi_u, optimize=True)
        grad_d = np.zeros((n_elem, 2, 2), dtype=float)
        grad_d[:, 0, :] = np.einsum("el,elk->ek", mx_k, grad_phi_mx, optimize=True)
        grad_d[:, 1, :] = np.einsum("el,elk->ek", my_k, grad_phi_my, optimize=True)
        grad_p = np.einsum("el,elk->ek", p_vals, grad_phi_p, optimize=True)

        F = np.zeros((n_elem, 2, 2), dtype=float)
        F[:, 0, 0] = 1.0 + grad_d[:, 0, 0]
        F[:, 0, 1] = grad_d[:, 0, 1]
        F[:, 1, 0] = grad_d[:, 1, 0]
        F[:, 1, 1] = 1.0 + grad_d[:, 1, 1]
        detF = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
        bad = np.argwhere(~np.isfinite(detF) | (np.abs(detF) <= 1.0e-14))
        if bad.size:
            raise RuntimeError(f"Singular ALE deformation gradient on element {int(bad[0, 0])}.")
        Finv = np.zeros_like(F)
        Finv[:, 0, 0] = F[:, 1, 1] / detF
        Finv[:, 0, 1] = -F[:, 0, 1] / detF
        Finv[:, 1, 0] = -F[:, 1, 0] / detF
        Finv[:, 1, 1] = F[:, 0, 0] / detF
        grad_u_phys = np.einsum("eij,ejk->eik", grad_u, Finv, optimize=True)
        grad_p_phys = np.einsum("eji,ej->ei", Finv, grad_p, optimize=True)
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
        resolved_conv_velocity = u_k_q - w_mesh_q
        if a_curr_q is None:
            current_acceleration_q = float(bossak["ma0"]) * (u_k_q - u_prev_q) + float(bossak["ma2"]) * a_prev_q
        else:
            current_acceleration_q = a_curr_q
        a_relaxed = current_acceleration_q
        if bool(use_oss):
            momentum_projection = np.asarray(state.momentum_projection, dtype=float).reshape(n_elem, n_q, 2)
        else:
            momentum_projection = np.zeros((n_elem, n_q, 2), dtype=float)
        old_subscale = np.asarray(state.old_subscale_velocity, dtype=float).reshape(n_elem, n_q, 2)
        static_residual = -(
            rho_value * a_relaxed
            + rho_value * np.einsum("eij,eqj->eqi", grad_u_phys, resolved_conv_velocity, optimize=True)
            + grad_p_phys[:, None, :]
            + momentum_projection
        ) + (rho_value / dt_value) * old_subscale

        h_e = _kratos_dvms_current_element_size_array(mesh, dh, d_mesh).reshape(-1)
        if h_e.shape[0] != n_elem:
            raise RuntimeError(f"DVMS predictor element-size mismatch: expected {n_elem}, got {h_e.shape[0]}.")
        predicted = np.asarray(state.predicted_subscale_velocity, dtype=float).reshape(n_elem, n_q, 2).copy()
        failed = np.zeros((n_elem, n_q), dtype=bool)
        converged = np.zeros_like(failed)
        max_it = max(int(max_iterations), 1)
        velocity_tol_value = max(float(rel_tol), 0.0)
        residual_tol_value = max(float(abs_tol), 0.0)
        grad_u_phys_q = np.broadcast_to(grad_u_phys[:, None, :, :], (n_elem, n_q, 2, 2))
        h_qp = np.broadcast_to(h_e[:, None], (n_elem, n_q))
        for _ in range(max_it):
            active = ~(failed | converged)
            if not np.any(active):
                break
            conv_velocity = resolved_conv_velocity + predicted
            conv_speed = np.linalg.norm(conv_velocity, axis=2)
            inv_tau = (
                (8.0 * mu_value / np.maximum(h_qp * h_qp, 1.0e-30))
                + rho_value * (1.0 / dt_value + 2.0 * conv_speed / np.maximum(h_qp, 1.0e-30))
            ) / dynamic_tau_value
            linearization = rho_value * grad_u_phys_q.copy()
            linearization[:, :, 0, 0] += inv_tau
            linearization[:, :, 1, 1] += inv_tau
            newton_rhs = static_residual - np.einsum("eqij,eqj->eqi", linearization, predicted, optimize=True)
            residual_norm_sq = np.einsum("eqi,eqi->eq", newton_rhs, newton_rhs, optimize=True)
            delta, ok = _solve_2x2_batched(linearization.reshape(-1, 2, 2), newton_rhs.reshape(-1, 2))
            delta = delta.reshape(n_elem, n_q, 2)
            ok = ok.reshape(n_elem, n_q)
            valid = active & ok & np.all(np.isfinite(delta), axis=2)
            invalid = active & ~valid
            if np.any(invalid):
                failed[invalid] = True
                predicted[invalid, :] = 0.0
            if not np.any(valid):
                continue
            predicted[valid, :] = predicted[valid, :] + delta[valid, :]
            delta_norm_sq = np.einsum("ij,ij->i", delta[valid, :], delta[valid, :], optimize=True)
            velocity_norm_sq = np.einsum(
                "ij,ij->i",
                predicted[valid, :],
                predicted[valid, :],
                optimize=True,
            )
            velocity_error = np.asarray(delta_norm_sq, dtype=float)
            use_relative = velocity_norm_sq > velocity_tol_value
            velocity_error[use_relative] = velocity_error[use_relative] / velocity_norm_sq[use_relative]
            residual_error = np.asarray(residual_norm_sq[valid], dtype=float)
            converged_now = (velocity_error <= velocity_tol_value) | (residual_error <= residual_tol_value)
            valid_idx = np.argwhere(valid)
            if valid_idx.size:
                converged[valid_idx[converged_now, 0], valid_idx[converged_now, 1]] = True
        remaining = ~(failed | converged)
        if np.any(remaining):
            # Kratos d_vms.cpp stores ZeroVector(Dim) at capped,
            # non-converged quadrature points.
            predicted[remaining, :] = 0.0
        state.predicted_subscale_velocity[:, :] = predicted.reshape(int(state.sample_count), 2)
        state.sync_coefficient("predicted_subscale_velocity")
        return

    backend_name = _normalized_dvms_operator_backend(backend)
    compiler = _quadrature_expression_compiler(state, dh, backend=backend_name)
    dt_value = max(float(dt), 1.0e-14)
    rho_value = float(rho_f)
    mu_value = float(mu_f)
    # Kratos' DVMS predictor ignores DYNAMIC_TAU and uses 1 / dt directly.
    # The argument is retained for API compatibility with older callers.
    dynamic_tau_value = 1.0
    bossak = _bossak_coefficients(alpha=float(bossak_alpha), dt=dt_value)
    momentum_projection_expr = (
        state.coefficient("momentum_projection")
        if bool(use_oss)
        else Constant(np.zeros((2,), dtype=float), dim=1)
    )
    predictor = build_fluid_dvms_predictor_symbolics(
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
        dt=dt_value,
        bossak_ma0=float(bossak["ma0"]),
        bossak_ma2=float(bossak["ma2"]),
        bossak_alpha=float(bossak["alpha"]),
        rho=rho_value,
        old_subscale=state.coefficient("old_subscale_velocity"),
        momentum_projection=momentum_projection_expr,
    )
    evaluated = compiler.evaluate_volume_expressions_on_quadrature(
        {
            "static_residual": predictor.static_residual,
            "grad_u_phys": predictor.kinematics.grad_u_phys,
            "resolved_conv_velocity": predictor.kinematics.resolved_conv_velocity,
        },
        layout=state.quadrature_layout,
        element_ids=_all_element_ids(state),
    )
    static_residual = np.asarray(evaluated["static_residual"], dtype=float).reshape(
        int(state.n_elements), int(state.n_qp_per_element), 2
    )
    grad_u_phys = np.asarray(evaluated["grad_u_phys"], dtype=float).reshape(
        int(state.n_elements), int(state.n_qp_per_element), 2, 2
    )
    resolved_conv_velocity = np.asarray(evaluated["resolved_conv_velocity"], dtype=float).reshape(
        int(state.n_elements), int(state.n_qp_per_element), 2
    )
    if (
        not np.all(np.isfinite(static_residual))
        or not np.all(np.isfinite(grad_u_phys))
        or not np.all(np.isfinite(resolved_conv_velocity))
    ):
        raise RuntimeError("DVMS predictor evaluation produced non-finite quadrature data.")

    h_e = _kratos_dvms_current_element_size_array(mesh, dh, d_mesh).reshape(-1)
    if h_e.shape[0] != int(state.n_elements):
        raise RuntimeError(
            f"DVMS predictor element-size mismatch: expected {state.n_elements}, got {h_e.shape[0]}."
        )
    h_qp = np.broadcast_to(h_e[:, None], (int(state.n_elements), int(state.n_qp_per_element)))
    predicted = np.asarray(state.predicted_subscale_velocity, dtype=float).reshape(
        int(state.n_elements), int(state.n_qp_per_element), 2
    ).copy()
    failed = np.zeros((int(state.n_elements), int(state.n_qp_per_element)), dtype=bool)
    converged = np.zeros_like(failed)
    eye = np.eye(2, dtype=float)

    def _solve_batched(mats: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if int(mats.shape[0]) == 0:
            return np.zeros_like(rhs, dtype=float), np.zeros((0,), dtype=bool)
        try:
            solved = np.asarray(np.linalg.solve(mats, rhs[..., None]), dtype=float)[..., 0]
            return solved, np.ones((mats.shape[0],), dtype=bool)
        except np.linalg.LinAlgError:
            out = np.zeros_like(rhs, dtype=float)
            ok = np.zeros((mats.shape[0],), dtype=bool)
            for idx in range(int(mats.shape[0])):
                try:
                    out[idx, :] = np.asarray(np.linalg.solve(mats[idx], rhs[idx]), dtype=float)
                    ok[idx] = True
                except np.linalg.LinAlgError:
                    ok[idx] = False
            return out, ok

    max_it = max(int(max_iterations), 1)
    velocity_tol_value = max(float(rel_tol), 0.0)
    residual_tol_value = max(float(abs_tol), 0.0)
    for _ in range(max_it):
        active = ~(failed | converged)
        if not np.any(active):
            break
        conv_velocity = resolved_conv_velocity + predicted
        conv_speed = np.linalg.norm(conv_velocity, axis=2)
        inv_tau = (
            (8.0 * mu_value / np.maximum(h_qp * h_qp, 1.0e-30))
            + rho_value * (1.0 / dt_value + 2.0 * conv_speed / np.maximum(h_qp, 1.0e-30))
        ) / dynamic_tau_value
        linearization = rho_value * grad_u_phys + inv_tau[..., None, None] * eye
        active_flat = np.flatnonzero(active.reshape(-1))
        pred_flat = predicted.reshape(-1, 2)
        linearization_flat = linearization.reshape(-1, 2, 2)
        static_flat = static_residual.reshape(-1, 2)
        rhs_flat = static_flat - np.einsum("nij,nj->ni", linearization_flat, pred_flat, optimize=True)
        residual_norm_sq_flat = np.einsum("ni,ni->n", rhs_flat, rhs_flat, optimize=True)
        solved_active, solved_ok = _solve_batched(
            linearization_flat[active_flat],
            rhs_flat[active_flat],
        )
        valid_active = solved_ok & np.all(np.isfinite(solved_active), axis=1)
        invalid_idx = active_flat[~valid_active]
        if invalid_idx.size:
            failed.reshape(-1)[invalid_idx] = True
            pred_flat[invalid_idx, :] = 0.0
        if not np.any(valid_active):
            continue
        solved_idx = active_flat[valid_active]
        delta = solved_active[valid_active]
        pred_flat[solved_idx, :] = pred_flat[solved_idx, :] + delta
        delta_norm_sq = np.einsum("ij,ij->i", delta, delta, optimize=True)
        velocity_norm_sq = np.einsum(
            "ij,ij->i",
            pred_flat[solved_idx, :],
            pred_flat[solved_idx, :],
            optimize=True,
        )
        velocity_error = np.asarray(delta_norm_sq, dtype=float)
        use_relative = velocity_norm_sq > velocity_tol_value
        velocity_error[use_relative] = velocity_error[use_relative] / velocity_norm_sq[use_relative]
        residual_error = np.asarray(residual_norm_sq_flat[solved_idx], dtype=float)
        converged_now = (velocity_error <= velocity_tol_value) | (residual_error <= residual_tol_value)
        converged.reshape(-1)[solved_idx[converged_now]] = True

    remaining = ~(failed | converged)
    if np.any(remaining):
        # Match DVMS::UpdateSubscaleVelocityPrediction:
        # mPredictedSubscaleVelocity = converged ? u : ZeroVector(Dim).
        predicted[remaining, :] = 0.0

    state.predicted_subscale_velocity[:, :] = predicted.reshape(int(state.sample_count), 2)
    state.sync_coefficient("predicted_subscale_velocity")


__all__ = [
    "_report_fluid_dvms_subscale_after_step",
    "_clear_fluid_dvms_oss_projections",
    "_update_fluid_dvms_oss_projections",
    "_update_fluid_dvms_old_subscale_after_step",
    "_update_fluid_dvms_predicted_subscale",
    "_update_fluid_dvms_state_from_previous_step",
]
