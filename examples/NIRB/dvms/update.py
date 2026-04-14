from __future__ import annotations

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import Function, VectorFunction

from .helpers import _bossak_coefficients, _field_values_on_global_dofs, _kratos_dvms_element_size_array
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
    backend: str | None = None,
) -> None:
    if int(state.sample_count) == 0:
        return
    fast = _dvms_fast_p1_tri_cache(state, dh, mesh)
    if fast is not None:
        ux_prev = _scalar_locals(dh, "ux", u_prev.components[0].nodal_values, fast["element_maps"]["ux"])
        uy_prev = _scalar_locals(dh, "uy", u_prev.components[1].nodal_values, fast["element_maps"]["uy"])
        mx_prev = _scalar_locals(dh, "mx", d_prev.components[0].nodal_values, fast["element_maps"]["mx"])
        my_prev = _scalar_locals(dh, "my", d_prev.components[1].nodal_values, fast["element_maps"]["my"])
        grad_phi_u = _grad_phi_phys(fast["grad_ref"]["ux"], fast["J_inv"])
        grad_phi_mx = _grad_phi_phys(fast["grad_ref"]["mx"], fast["J_inv"])
        grad_phi_my = _grad_phi_phys(fast["grad_ref"]["my"], fast["J_inv"])
        grad_ux = np.einsum("el,elk->ek", ux_prev, grad_phi_u, optimize=True)
        grad_uy = np.einsum("el,elk->ek", uy_prev, grad_phi_u, optimize=True)
        grad_mx = np.einsum("el,elk->ek", mx_prev, grad_phi_mx, optimize=True)
        grad_my = np.einsum("el,elk->ek", my_prev, grad_phi_my, optimize=True)
        F_old = np.zeros((int(state.n_elements), 2, 2), dtype=float)
        F_old[:, 0, 0] = 1.0 + grad_mx[:, 0]
        F_old[:, 0, 1] = grad_mx[:, 1]
        F_old[:, 1, 0] = grad_my[:, 0]
        F_old[:, 1, 1] = 1.0 + grad_my[:, 1]
        detF = F_old[:, 0, 0] * F_old[:, 1, 1] - F_old[:, 0, 1] * F_old[:, 1, 0]
        bad = np.argwhere(~np.isfinite(detF) | (np.abs(detF) <= 1.0e-14))
        if bad.size:
            raise RuntimeError(f"Singular previous ALE deformation gradient on element {int(bad[0, 0])}.")
        cof = np.zeros_like(F_old)
        cof[:, 0, 0] = F_old[:, 1, 1]
        cof[:, 0, 1] = -F_old[:, 0, 1]
        cof[:, 1, 0] = -F_old[:, 1, 0]
        cof[:, 1, 1] = F_old[:, 0, 0]
        grad_u = np.zeros_like(F_old)
        grad_u[:, 0, :] = grad_ux
        grad_u[:, 1, :] = grad_uy
        old_mass = -np.einsum("eij,eij->e", cof, grad_u, optimize=True) / detF
        state._reshape_scalar_quadrature(state.old_mass_residual)[...] = old_mass[:, None]
        state.sync_coefficient("old_mass_residual")
        return

    compiler = _quadrature_expression_compiler(state, dh, backend=backend)
    expr = build_fluid_dvms_old_mass_residual(u_prev=u_prev, d_prev=d_prev)
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


def _update_fluid_dvms_predicted_subscale(
    *,
    state: FluidDVMSState,
    dh: DofHandler,
    mesh: Mesh,
    u_k: VectorFunction,
    u_prev: VectorFunction,
    a_prev: VectorFunction,
    p_k: Function,
    d_mesh: VectorFunction,
    d_prev: VectorFunction,
    d_prev2: VectorFunction | None = None,
    mesh_v_prev: VectorFunction | None = None,
    mesh_a_prev: VectorFunction | None = None,
    rho_f: float,
    mu_f: float,
    dt: float,
    bossak_alpha: float,
    dynamic_tau: float,
    max_iterations: int = 10,
    rel_tol: float = 1.0e-14,
    abs_tol: float = 1.0e-14,
    backend: str | None = None,
) -> None:
    if int(state.sample_count) == 0:
        return
    fast = _dvms_fast_p1_tri_cache(state, dh, mesh)
    if fast is not None:
        dt_value = max(float(dt), 1.0e-14)
        rho_value = float(rho_f)
        mu_value = float(mu_f)
        dynamic_tau_value = max(float(dynamic_tau), 1.0e-14)
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
        p_vals = _scalar_locals(dh, "p", p_k.nodal_values, fast["element_maps"]["p"])
        mx_k = _scalar_locals(dh, "mx", d_mesh.components[0].nodal_values, fast["element_maps"]["mx"])
        my_k = _scalar_locals(dh, "my", d_mesh.components[1].nodal_values, fast["element_maps"]["my"])
        mx_prev = _scalar_locals(dh, "mx", d_prev.components[0].nodal_values, fast["element_maps"]["mx"])
        my_prev = _scalar_locals(dh, "my", d_prev.components[1].nodal_values, fast["element_maps"]["my"])
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
        if mesh_v_prev is not None and mesh_a_prev is not None:
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
        a_curr = float(bossak["ma0"]) * (u_k_q - u_prev_q) + float(bossak["ma2"]) * a_prev_q
        a_relaxed = (1.0 - float(bossak["alpha"])) * a_curr + float(bossak["alpha"]) * a_prev_q
        momentum_projection = np.asarray(state.momentum_projection, dtype=float).reshape(n_elem, n_q, 2)
        old_subscale = np.asarray(state.old_subscale_velocity, dtype=float).reshape(n_elem, n_q, 2)
        static_residual = -(
            rho_value * a_relaxed
            + rho_value * np.einsum("eij,eqj->eqi", grad_u_phys, resolved_conv_velocity, optimize=True)
            + grad_p_phys[:, None, :]
            + momentum_projection
        ) + (rho_value / dt_value) * old_subscale

        h_e = _kratos_dvms_element_size_array(mesh).reshape(-1)
        if h_e.shape[0] != n_elem:
            raise RuntimeError(f"DVMS predictor element-size mismatch: expected {n_elem}, got {h_e.shape[0]}.")
        predicted = np.asarray(state.predicted_subscale_velocity, dtype=float).reshape(n_elem, n_q, 2).copy()
        failed = np.zeros((n_elem, n_q), dtype=bool)
        converged = np.zeros_like(failed)
        max_it = max(int(max_iterations), 1)
        rel_tol_value = float(rel_tol)
        abs_tol_value = float(abs_tol)
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
            solved, ok = _solve_2x2_batched(linearization.reshape(-1, 2, 2), static_residual.reshape(-1, 2))
            solved = solved.reshape(n_elem, n_q, 2)
            ok = ok.reshape(n_elem, n_q)
            valid = active & ok & np.all(np.isfinite(solved), axis=2)
            invalid = active & ~valid
            if np.any(invalid):
                failed[invalid] = True
                predicted[invalid, :] = 0.0
            if not np.any(valid):
                continue
            delta = solved[valid, :] - predicted[valid, :]
            predicted[valid, :] = solved[valid, :]
            err = np.linalg.norm(delta, axis=1)
            norm_u = np.linalg.norm(predicted[valid, :], axis=1)
            rel_err = np.divide(err, norm_u, out=np.full_like(err, np.inf, dtype=float), where=norm_u > 0.0)
            converged_now = (err <= abs_tol_value) | ((norm_u > rel_tol_value) & (rel_err <= rel_tol_value))
            valid_idx = np.argwhere(valid)
            if valid_idx.size:
                converged[valid_idx[converged_now, 0], valid_idx[converged_now, 1]] = True
        remaining = ~(failed | converged)
        if np.any(remaining):
            predicted[remaining, :] = 0.0
        state.predicted_subscale_velocity[:, :] = predicted.reshape(int(state.sample_count), 2)
        state.sync_coefficient("predicted_subscale_velocity")
        return

    backend_name = _normalized_dvms_operator_backend(backend)
    compiler = _quadrature_expression_compiler(state, dh, backend=backend_name)
    dt_value = max(float(dt), 1.0e-14)
    rho_value = float(rho_f)
    mu_value = float(mu_f)
    dynamic_tau_value = max(float(dynamic_tau), 1.0e-14)
    bossak = _bossak_coefficients(alpha=float(bossak_alpha), dt=dt_value)
    predictor = build_fluid_dvms_predictor_symbolics(
        u_k=u_k,
        u_prev=u_prev,
        a_prev=a_prev,
        p_k=p_k,
        d_mesh=d_mesh,
        d_prev=d_prev,
        d_prev2=d_prev2,
        mesh_v_prev=mesh_v_prev,
        mesh_a_prev=mesh_a_prev,
        dt=dt_value,
        bossak_ma0=float(bossak["ma0"]),
        bossak_ma2=float(bossak["ma2"]),
        bossak_alpha=float(bossak["alpha"]),
        rho=rho_value,
        old_subscale=state.coefficient("old_subscale_velocity"),
        momentum_projection=state.coefficient("momentum_projection"),
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

    h_e = _kratos_dvms_element_size_array(mesh).reshape(-1)
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
    rel_tol_value = float(rel_tol)
    abs_tol_value = float(abs_tol)
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
        solved_active, solved_ok = _solve_batched(
            linearization_flat[active_flat],
            static_flat[active_flat],
        )
        valid_active = solved_ok & np.all(np.isfinite(solved_active), axis=1)
        invalid_idx = active_flat[~valid_active]
        if invalid_idx.size:
            failed.reshape(-1)[invalid_idx] = True
            pred_flat[invalid_idx, :] = 0.0
        if not np.any(valid_active):
            continue
        solved_idx = active_flat[valid_active]
        delta = solved_active[valid_active] - pred_flat[solved_idx, :]
        pred_flat[solved_idx, :] = solved_active[valid_active]
        err = np.linalg.norm(delta, axis=1)
        norm_u = np.linalg.norm(pred_flat[solved_idx, :], axis=1)
        rel_err = np.divide(
            err,
            norm_u,
            out=np.full_like(err, np.inf, dtype=float),
            where=norm_u > 0.0,
        )
        converged_now = (err <= abs_tol_value) | (
            (norm_u > rel_tol_value) & (rel_err <= rel_tol_value)
        )
        converged.reshape(-1)[solved_idx[converged_now]] = True

    remaining = ~(failed | converged)
    if np.any(remaining):
        predicted[remaining, :] = 0.0

    state.predicted_subscale_velocity[:, :] = predicted.reshape(int(state.sample_count), 2)
    state.sync_coefficient("predicted_subscale_velocity")


__all__ = [
    "_update_fluid_dvms_predicted_subscale",
    "_update_fluid_dvms_state_from_previous_step",
]
