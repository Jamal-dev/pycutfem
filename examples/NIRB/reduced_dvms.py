from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping

import numpy as np

from examples.NIRB.dvms.helpers import _triangle2d3_minimum_element_size
from examples.NIRB.dvms.update import _dvms_fast_p1_tri_cache, _grad_phi_phys

try:  # pragma: no cover - exercised when numba is available in the runtime env
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


_SINGULAR_TOL = 1.0e-14


def _env_flag(name: str) -> bool:
    value = str(os.getenv(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _reduced_dvms_cpp_enabled() -> bool:
    return _env_flag("PYCUTFEM_NIRB_REDUCED_CPP") or _env_flag("PYCUTFEM_NIRB_REDUCED_CPP_REQUIRED")


def _reduced_dvms_cpp_required() -> bool:
    return _env_flag("PYCUTFEM_NIRB_REDUCED_CPP_REQUIRED")


def _cpp_dvms_module():
    from examples.NIRB.cpp_backend.reduced_dvms import module as _module

    return _module()


def reduced_dvms_cpp_backend_status() -> dict[str, object]:
    enabled = _reduced_dvms_cpp_enabled()
    required = _reduced_dvms_cpp_required()
    try:
        module = _cpp_dvms_module()
    except Exception as exc:  # pragma: no cover - depends on local compiler/toolchain
        if required:
            raise RuntimeError("PYCUTFEM_NIRB_REDUCED_CPP_REQUIRED=1 but the C++ DVMS backend failed.") from exc
        return {
            "enabled": bool(enabled),
            "required": bool(required),
            "loaded": False,
            "backend": "python",
            "unavailable_reason": str(exc),
        }
    return {
        "enabled": bool(enabled),
        "required": bool(required),
        "loaded": True,
        "backend": "cpp" if enabled else "available",
        "module": getattr(module, "__name__", type(module).__name__),
        "module_file": getattr(module, "__file__", None),
    }


@dataclass
class ReducedDVMSState:
    """Array-only DVMS state sampled on element quadrature points.

    All fields are stored by element and quadrature point. Vector fields use
    shape ``(n_elements, n_q, 2)``; scalar fields use ``(n_elements, n_q)``.
    """

    predicted_subscale_velocity: np.ndarray
    old_subscale_velocity: np.ndarray
    momentum_projection: np.ndarray
    mass_projection: np.ndarray
    old_mass_residual: np.ndarray

    def __post_init__(self) -> None:
        self.predicted_subscale_velocity = self._as_vector(
            self.predicted_subscale_velocity,
            "predicted_subscale_velocity",
        )
        base_shape = self.predicted_subscale_velocity.shape[:2]
        self.old_subscale_velocity = self._as_vector(
            self.old_subscale_velocity,
            "old_subscale_velocity",
            base_shape=base_shape,
        )
        self.momentum_projection = self._as_vector(
            self.momentum_projection,
            "momentum_projection",
            base_shape=base_shape,
        )
        self.mass_projection = self._as_scalar(
            self.mass_projection,
            "mass_projection",
            base_shape=base_shape,
        )
        self.old_mass_residual = self._as_scalar(
            self.old_mass_residual,
            "old_mass_residual",
            base_shape=base_shape,
        )

    @property
    def n_elements(self) -> int:
        return int(self.predicted_subscale_velocity.shape[0])

    @property
    def n_q(self) -> int:
        return int(self.predicted_subscale_velocity.shape[1])

    def select(self, elements: np.ndarray | list[int] | tuple[int, ...], *, copy: bool = True) -> "ReducedDVMSState":
        """Return a state containing only the requested element blocks."""

        ids = np.asarray(elements, dtype=int).reshape(-1)
        if np.any(ids < 0) or np.any(ids >= self.n_elements):
            raise ValueError("elements contain out-of-range entries")

        def take(values: np.ndarray) -> np.ndarray:
            out = np.asarray(values)[ids]
            return out.copy() if copy else out

        return ReducedDVMSState(
            predicted_subscale_velocity=take(self.predicted_subscale_velocity),
            old_subscale_velocity=take(self.old_subscale_velocity),
            momentum_projection=take(self.momentum_projection),
            mass_projection=take(self.mass_projection),
            old_mass_residual=take(self.old_mass_residual),
        )

    @staticmethod
    def _as_vector(
        values: np.ndarray,
        name: str,
        *,
        base_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 3 or arr.shape[2] != 2:
            raise ValueError(f"{name} must have shape (n_elements, n_q, 2), got {arr.shape}")
        if base_shape is not None and arr.shape[:2] != base_shape:
            raise ValueError(f"{name} must have leading shape {base_shape}, got {arr.shape[:2]}")
        return arr

    @staticmethod
    def _as_scalar(
        values: np.ndarray,
        name: str,
        *,
        base_shape: tuple[int, int],
    ) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.shape != base_shape:
            raise ValueError(f"{name} must have shape {base_shape}, got {arr.shape}")
        return arr


def update_predicted_subscale_local(
    *,
    predicted_subscale_velocity: np.ndarray,
    old_subscale_velocity: np.ndarray,
    u_k_q: np.ndarray,
    u_prev_q: np.ndarray,
    a_prev_q: np.ndarray,
    p_grad_phys: np.ndarray,
    grad_u_phys: np.ndarray,
    h_e: np.ndarray,
    rho_f: float,
    mu_f: float,
    dt: float,
    bossak_alpha: float,
    a_curr_q: np.ndarray | None = None,
    momentum_projection: np.ndarray | None = None,
    resolved_conv_velocity: np.ndarray | None = None,
    mesh_v_q: np.ndarray | None = None,
    d_mesh_q: np.ndarray | None = None,
    d_prev_q: np.ndarray | None = None,
    d_prev2_q: np.ndarray | None = None,
    mesh_v_prev_q: np.ndarray | None = None,
    mesh_a_prev_q: np.ndarray | None = None,
    max_iterations: int = 10,
    rel_tol: float = 1.0e-14,
    abs_tol: float = 1.0e-14,
) -> np.ndarray:
    """Return the reduced local predicted-subscale update.

    Inputs are already decoded at quadrature points. Provide either
    ``resolved_conv_velocity`` directly or enough mesh data to form it:
    ``mesh_v_q``; or Bossak mesh velocity data; or ``d_mesh_q``/``d_prev_q``
    with optional ``d_prev2_q`` for the BDF2 fallback used by the full path.
    Singular solves, non-finite solves, and non-converged quadrature points are
    returned as zero vectors, matching the fast P1-triangle DVMS branch.
    """

    predicted = _as_vector(predicted_subscale_velocity, "predicted_subscale_velocity").copy()
    base_shape = predicted.shape[:2]
    old_subscale = _as_vector(old_subscale_velocity, "old_subscale_velocity", base_shape=base_shape)
    u_k = _as_vector(u_k_q, "u_k_q", base_shape=base_shape)
    u_prev = _as_vector(u_prev_q, "u_prev_q", base_shape=base_shape)
    a_prev = _as_vector(a_prev_q, "a_prev_q", base_shape=base_shape)
    grad_p = _as_quadrature_vector(p_grad_phys, "p_grad_phys", base_shape)
    grad_u = _as_grad_u(grad_u_phys, "grad_u_phys", base_shape)
    h_qp = _as_h_qp(h_e, base_shape)

    if a_curr_q is None:
        coeffs = _bossak_coefficients(alpha=float(bossak_alpha), dt=max(float(dt), _SINGULAR_TOL))
        acceleration = float(coeffs["ma0"]) * (u_k - u_prev) + float(coeffs["ma2"]) * a_prev
    else:
        acceleration = _as_vector(a_curr_q, "a_curr_q", base_shape=base_shape)

    if momentum_projection is None:
        momentum = np.zeros(base_shape + (2,), dtype=float)
    else:
        momentum = _as_vector(momentum_projection, "momentum_projection", base_shape=base_shape)

    dt_value = max(float(dt), _SINGULAR_TOL)
    rho_value = float(rho_f)
    mu_value = float(mu_f)
    conv_resolved = _resolved_conv_velocity(
        base_shape=base_shape,
        u_k_q=u_k,
        dt=dt_value,
        bossak_alpha=float(bossak_alpha),
        resolved_conv_velocity=resolved_conv_velocity,
        mesh_v_q=mesh_v_q,
        d_mesh_q=d_mesh_q,
        d_prev_q=d_prev_q,
        d_prev2_q=d_prev2_q,
        mesh_v_prev_q=mesh_v_prev_q,
        mesh_a_prev_q=mesh_a_prev_q,
    )
    static_residual = -(
        rho_value * acceleration
        + rho_value * np.einsum("eqij,eqj->eqi", grad_u, conv_resolved, optimize=True)
        + grad_p
        + momentum
    ) + (rho_value / dt_value) * old_subscale

    failed = np.zeros(base_shape, dtype=bool)
    converged = np.zeros(base_shape, dtype=bool)
    max_it = max(int(max_iterations), 1)
    velocity_tol_value = max(float(rel_tol), 0.0)
    residual_tol_value = max(float(abs_tol), 0.0)
    h_safe = np.maximum(h_qp, 1.0e-30)

    for _ in range(max_it):
        active = ~(failed | converged)
        if not np.any(active):
            break
        conv_velocity = conv_resolved + predicted
        conv_speed = np.linalg.norm(conv_velocity, axis=2)
        inv_tau = (8.0 * mu_value / np.maximum(h_safe * h_safe, 1.0e-30)) + rho_value * (
            1.0 / dt_value + 2.0 * conv_speed / h_safe
        )
        linearization = rho_value * grad_u.copy()
        linearization[:, :, 0, 0] += inv_tau
        linearization[:, :, 1, 1] += inv_tau
        rhs = static_residual - np.einsum("eqij,eqj->eqi", linearization, predicted, optimize=True)
        residual_norm_sq = np.einsum("eqi,eqi->eq", rhs, rhs, optimize=True)
        delta, ok = _solve_2x2_batched(linearization.reshape(-1, 2, 2), rhs.reshape(-1, 2))
        delta = delta.reshape(base_shape + (2,))
        ok = ok.reshape(base_shape)
        valid = active & ok & np.all(np.isfinite(delta), axis=2)
        invalid = active & ~valid
        if np.any(invalid):
            failed[invalid] = True
            predicted[invalid, :] = 0.0
        if not np.any(valid):
            continue
        predicted[valid, :] = predicted[valid, :] + delta[valid, :]
        delta_norm_sq = np.einsum("ij,ij->i", delta[valid, :], delta[valid, :], optimize=True)
        velocity_norm_sq = np.einsum("ij,ij->i", predicted[valid, :], predicted[valid, :], optimize=True)
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
        predicted[remaining, :] = 0.0
    return predicted


def _resolved_conv_velocity(
    *,
    base_shape: tuple[int, int],
    u_k_q: np.ndarray,
    dt: float,
    bossak_alpha: float,
    resolved_conv_velocity: np.ndarray | None,
    mesh_v_q: np.ndarray | None,
    d_mesh_q: np.ndarray | None,
    d_prev_q: np.ndarray | None,
    d_prev2_q: np.ndarray | None,
    mesh_v_prev_q: np.ndarray | None,
    mesh_a_prev_q: np.ndarray | None,
) -> np.ndarray:
    if resolved_conv_velocity is not None:
        return _as_vector(resolved_conv_velocity, "resolved_conv_velocity", base_shape=base_shape)
    if mesh_v_q is not None:
        return u_k_q - _as_vector(mesh_v_q, "mesh_v_q", base_shape=base_shape)
    if d_mesh_q is None or d_prev_q is None:
        raise ValueError("resolved_conv_velocity or mesh velocity/displacement arrays are required")

    d_mesh = _as_vector(d_mesh_q, "d_mesh_q", base_shape=base_shape)
    d_prev = _as_vector(d_prev_q, "d_prev_q", base_shape=base_shape)
    if mesh_v_prev_q is not None and mesh_a_prev_q is not None:
        mesh_v_prev = _as_vector(mesh_v_prev_q, "mesh_v_prev_q", base_shape=base_shape)
        mesh_a_prev = _as_vector(mesh_a_prev_q, "mesh_a_prev_q", base_shape=base_shape)
        coeffs = _bossak_coefficients(alpha=bossak_alpha, dt=dt)
        beta = float(coeffs["beta"])
        gamma = float(coeffs["gamma"])
        a_mesh = (d_mesh - d_prev - dt * mesh_v_prev - dt * dt * (0.5 - beta) * mesh_a_prev) / (beta * dt * dt)
        w_mesh = mesh_v_prev + dt * ((1.0 - gamma) * mesh_a_prev + gamma * a_mesh)
        return u_k_q - w_mesh

    if d_prev2_q is None:
        d_prev2 = np.zeros(base_shape + (2,), dtype=float)
    else:
        d_prev2 = _as_vector(d_prev2_q, "d_prev2_q", base_shape=base_shape)
    w_mesh = (1.5 * d_mesh - 2.0 * d_prev + 0.5 * d_prev2) / dt
    return u_k_q - w_mesh


def _bossak_coefficients(*, alpha: float, dt: float) -> dict[str, float]:
    alpha_value = float(alpha)
    dt_value = max(float(dt), _SINGULAR_TOL)
    gamma = 0.5 - alpha_value
    if gamma <= 0.0:
        raise ValueError(f"Bossak alpha={alpha_value} yields non-positive gamma={gamma}.")
    beta = 0.25 * (1.0 - alpha_value) * (1.0 - alpha_value)
    return {
        "alpha": alpha_value,
        "gamma": gamma,
        "beta": beta,
        "ma0": 1.0 / (gamma * dt_value),
        "ma2": (-1.0 + gamma) / gamma,
    }


def _as_vector(
    values: np.ndarray,
    name: str,
    *,
    base_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(f"{name} must have shape (n_elements, n_q, 2), got {arr.shape}")
    if base_shape is not None and arr.shape[:2] != base_shape:
        raise ValueError(f"{name} must have leading shape {base_shape}, got {arr.shape[:2]}")
    return arr


def _as_grad_u(values: np.ndarray, name: str, base_shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape == (base_shape[0], 2, 2):
        return np.broadcast_to(arr[:, None, :, :], base_shape + (2, 2))
    if arr.shape == base_shape + (2, 2):
        return arr
    raise ValueError(f"{name} must have shape (n_elements, 2, 2) or (n_elements, n_q, 2, 2), got {arr.shape}")


def _as_quadrature_vector(values: np.ndarray, name: str, base_shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape == (base_shape[0], 2):
        return np.broadcast_to(arr[:, None, :], base_shape + (2,))
    return _as_vector(arr, name, base_shape=base_shape)


def _as_h_qp(values: np.ndarray, base_shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape == (base_shape[0],):
        return np.broadcast_to(arr[:, None], base_shape)
    if arr.shape == base_shape:
        return arr
    raise ValueError(f"h_e must have shape ({base_shape[0]},) or {base_shape}, got {arr.shape}")


def _solve_2x2_batched(A: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    ok = (
        np.isfinite(det)
        & (np.abs(det) > _SINGULAR_TOL)
        & np.all(np.isfinite(A), axis=(-2, -1))
        & np.all(np.isfinite(rhs), axis=-1)
    )
    out = np.zeros_like(rhs, dtype=float)
    if np.any(ok):
        A_ok = A[ok]
        rhs_ok = rhs[ok]
        det_ok = det[ok]
        out_ok = np.zeros_like(rhs_ok, dtype=float)
        out_ok[:, 0] = (A_ok[:, 1, 1] * rhs_ok[:, 0] - A_ok[:, 0, 1] * rhs_ok[:, 1]) / det_ok
        out_ok[:, 1] = (A_ok[:, 0, 0] * rhs_ok[:, 1] - A_ok[:, 1, 0] * rhs_ok[:, 0]) / det_ok
        out[ok, :] = out_ok
    return out, ok


def _local_grouped_gdofs_batch(dh, element_ids: np.ndarray) -> np.ndarray:
    eids = np.asarray(element_ids, dtype=int).reshape(-1)
    ux_map = np.asarray(dh.element_maps["ux"], dtype=int)[eids]
    uy_map = np.asarray(dh.element_maps["uy"], dtype=int)[eids]
    p_map = np.asarray(dh.element_maps["p"], dtype=int)[eids]
    return np.concatenate([ux_map, uy_map, p_map], axis=1).astype(int, copy=False)


def _field_local_array(
    field_locals: Mapping[str, np.ndarray],
    *,
    names: tuple[str, ...],
    field_name: str,
    element_maps: Mapping[str, np.ndarray],
    all_element_maps: Mapping[str, np.ndarray],
    element_ids: np.ndarray,
) -> np.ndarray:
    expected = np.asarray(element_maps[field_name], dtype=int).shape
    full_expected = np.asarray(all_element_maps[field_name], dtype=int).shape
    for name in names:
        if name not in field_locals:
            continue
        values = np.asarray(field_locals[name], dtype=float)
        if values.shape == full_expected and values.shape != expected:
            values = values[np.asarray(element_ids, dtype=int)]
        if values.shape != expected:
            raise ValueError(
                f"field_locals[{name!r}] must have shape {expected} or full element shape "
                f"{full_expected}, got {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"field_locals[{name!r}] contains non-finite values.")
        return values
    raise KeyError(f"Missing local field data for {names[0]!r}.")


def _current_element_size_from_local_mesh(
    mesh,
    *,
    element_ids: np.ndarray,
    mx: np.ndarray,
    my: np.ndarray,
) -> np.ndarray:
    eids = np.asarray(element_ids, dtype=int).reshape(-1)
    conn = np.asarray(mesh.elements_connectivity, dtype=int)[eids]
    coords = np.asarray(mesh.nodes_x_y_pos, dtype=float)[conn].copy()
    if coords.ndim != 3 or coords.shape[1:] != (3, 2):
        areas = np.asarray(mesh.areas_list, dtype=float).reshape(-1)[eids]
        return np.sqrt(np.maximum(np.abs(areas), 1.0e-30))
    coords[:, :, 0] += np.asarray(mx, dtype=float)
    coords[:, :, 1] += np.asarray(my, dtype=float)
    return _triangle2d3_minimum_element_size(coords)


def _assemble_kratos_system_core_impl(
    basis_u: np.ndarray,
    basis_p: np.ndarray,
    detJ_geo: np.ndarray,
    ref_weights: np.ndarray,
    detF: np.ndarray,
    cof: np.ndarray,
    grad_phi_u_ref: np.ndarray,
    grad_phi_u_cur: np.ndarray,
    grad_phi_p_cur: np.ndarray,
    grad_u_phys: np.ndarray,
    resolved_conv: np.ndarray,
    predicted: np.ndarray,
    old_subscale: np.ndarray,
    momentum_projection: np.ndarray,
    mass_projection: np.ndarray,
    old_mass_residual: np.ndarray,
    h_e: np.ndarray,
    x_local: np.ndarray,
    a_local: np.ndarray,
    body: np.ndarray,
    rho: float,
    mu: float,
    inv_dt: float,
    inc_scale: float,
    mam: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_elem = int(x_local.shape[0])
    n_q = int(ref_weights.shape[0])
    K_nonvisc = np.zeros((n_elem, 9, 9), dtype=np.float64)
    K_visc = np.zeros((n_elem, 9, 9), dtype=np.float64)
    M_elem = np.zeros((n_elem, 9, 9), dtype=np.float64)
    raw_rhs = np.zeros((n_elem, 9), dtype=np.float64)

    for e in range(n_elem):
        div_u = grad_u_phys[e, 0, 0] + grad_u_phys[e, 1, 1]
        visc_sigma = np.zeros((2, 2), dtype=np.float64)
        for r in range(2):
            for s in range(2):
                delta = 1.0 if r == s else 0.0
                visc_sigma[r, s] = mu * (
                    grad_u_phys[e, r, s] + grad_u_phys[e, s, r] - (2.0 / 3.0) * div_u * delta
                )
        visc_action = np.zeros((2, 2), dtype=np.float64)
        for r in range(2):
            for t in range(2):
                visc_action[r, t] = visc_sigma[r, 0] * cof[e, 0, t] + visc_sigma[r, 1] * cof[e, 1, t]

        for q in range(n_q):
            wgeo = detJ_geo[e] * ref_weights[q]
            weight = wgeo * detF[e]
            conv0 = resolved_conv[e, q, 0] + predicted[e, q, 0]
            conv1 = resolved_conv[e, q, 1] + predicted[e, q, 1]
            conv_speed = (conv0 * conv0 + conv1 * conv1) ** 0.5
            h_value = h_e[e]
            h_safe = h_value if h_value > 1.0e-30 else 1.0e-30
            tau_one = 1.0 / (
                8.0 * mu / (h_safe * h_safe)
                + rho * (inv_dt + 2.0 * conv_speed / h_safe)
            )
            tau_two = mu + rho * conv_speed * h_value / 4.0
            tau_p = rho * h_value * h_value * inv_dt / 8.0
            old_uss0 = rho * inv_dt * old_subscale[e, q, 0]
            old_uss1 = rho * inv_dt * old_subscale[e, q, 1]
            source0 = body[0] - momentum_projection[e, q, 0] + old_uss0
            source1 = body[1] - momentum_projection[e, q, 1] + old_uss1
            body_old0 = body[0] + old_uss0
            body_old1 = body[1] + old_uss1

            gdot = np.zeros(3, dtype=np.float64)
            for i in range(3):
                gdot[i] = grad_phi_u_cur[e, i, 0] * conv0 + grad_phi_u_cur[e, i, 1] * conv1

            for c in range(2):
                body_old_c = body_old0 if c == 0 else body_old1
                source_c = source0 if c == 0 else source1
                for i in range(3):
                    row = c * 3 + i
                    phi_i = basis_u[q, i]
                    div_v_cur = grad_phi_u_cur[e, i, c]
                    div_v_cof = cof[e, c, 0] * grad_phi_u_ref[e, i, 0] + cof[e, c, 1] * grad_phi_u_ref[e, i, 1]
                    tau_source_i = rho * (gdot[i] - inv_dt * phi_i)
                    tau_mass_i = rho * gdot[i] - inv_dt * phi_i

                    raw_rhs[e, row] += weight * phi_i * body_old_c
                    raw_rhs[e, row] += weight * tau_one * tau_source_i * source_c
                    raw_rhs[e, row] -= weight * inc_scale * (tau_two + tau_p) * mass_projection[e, q] * div_v_cur
                    raw_rhs[e, row] -= weight * inc_scale * tau_p * old_mass_residual[e, q] * div_v_cur
                    raw_rhs[e, row] -= wgeo * (
                        visc_action[c, 0] * grad_phi_u_ref[e, i, 0]
                        + visc_action[c, 1] * grad_phi_u_ref[e, i, 1]
                    )

                    for a in range(2):
                        for j in range(3):
                            col = a * 3 + j
                            if a == c:
                                K_nonvisc[e, row, col] += weight * rho * gdot[j] * phi_i
                                K_nonvisc[e, row, col] += weight * tau_one * tau_source_i * rho * gdot[j]
                                M_elem[e, row, col] += weight * rho * phi_i * basis_u[q, j]
                                M_elem[e, row, col] += weight * tau_one * tau_mass_i * rho * basis_u[q, j]
                            K_nonvisc[e, row, col] += (
                                weight
                                * inc_scale
                                * (tau_two + tau_p)
                                * div_v_cur
                                * grad_phi_u_cur[e, j, a]
                            )

                            grad_du = np.zeros((2, 2), dtype=np.float64)
                            grad_du[a, 0] = grad_phi_u_cur[e, j, 0]
                            grad_du[a, 1] = grad_phi_u_cur[e, j, 1]
                            div_du = grad_phi_u_cur[e, j, a]
                            sigma_du = np.zeros((2, 2), dtype=np.float64)
                            for r in range(2):
                                for s in range(2):
                                    delta = 1.0 if r == s else 0.0
                                    sigma_du[r, s] = mu * (
                                        grad_du[r, s] + grad_du[s, r] - (2.0 / 3.0) * div_du * delta
                                    )
                            action0 = sigma_du[c, 0] * cof[e, 0, 0] + sigma_du[c, 1] * cof[e, 1, 0]
                            action1 = sigma_du[c, 0] * cof[e, 0, 1] + sigma_du[c, 1] * cof[e, 1, 1]
                            K_visc[e, row, col] += wgeo * (
                                action0 * grad_phi_u_ref[e, i, 0] + action1 * grad_phi_u_ref[e, i, 1]
                            )

                    for j in range(3):
                        colp = 6 + j
                        K_nonvisc[e, row, colp] += weight * tau_one * tau_source_i * grad_phi_p_cur[e, j, c]
                        K_nonvisc[e, row, colp] -= wgeo * div_v_cof * basis_p[q, j]

            for i in range(3):
                row = 6 + i
                grad_q0 = grad_phi_p_cur[e, i, 0]
                grad_q1 = grad_phi_p_cur[e, i, 1]
                raw_rhs[e, row] += weight * inc_scale * tau_one * (grad_q0 * source0 + grad_q1 * source1)
                for a in range(2):
                    grad_qa = grad_q0 if a == 0 else grad_q1
                    for j in range(3):
                        col = a * 3 + j
                        K_nonvisc[e, row, col] += weight * inc_scale * tau_one * grad_qa * rho * gdot[j]
                        K_nonvisc[e, row, col] += (
                            wgeo
                            * basis_p[q, i]
                            * (cof[e, a, 0] * grad_phi_u_ref[e, j, 0] + cof[e, a, 1] * grad_phi_u_ref[e, j, 1])
                        )
                        M_elem[e, row, col] += weight * tau_one * grad_qa * rho * basis_u[q, j]
                for j in range(3):
                    colp = 6 + j
                    K_nonvisc[e, row, colp] += (
                        weight
                        * inc_scale
                        * tau_one
                        * (grad_q0 * grad_phi_p_cur[e, j, 0] + grad_q1 * grad_phi_p_cur[e, j, 1])
                    )

    K_elem = np.zeros((n_elem, 9, 9), dtype=np.float64)
    for e in range(n_elem):
        for i in range(9):
            kx = 0.0
            ma = 0.0
            for j in range(9):
                kx += K_nonvisc[e, i, j] * x_local[e, j]
                ma += M_elem[e, i, j] * a_local[e, j]
                K_elem[e, i, j] = K_nonvisc[e, i, j] + K_visc[e, i, j] + mam * M_elem[e, i, j]
            raw_rhs[e, i] = -(raw_rhs[e, i] - kx - ma)
    return K_elem, raw_rhs


_assemble_kratos_system_core_python = (
    njit(cache=True)(_assemble_kratos_system_core_impl) if njit is not None else _assemble_kratos_system_core_impl
)


def _assemble_kratos_system_core(*args) -> tuple[np.ndarray, np.ndarray]:
    if _reduced_dvms_cpp_enabled():
        try:
            K_elem, raw_rhs = _cpp_dvms_module().assemble_kratos_system_core(*args)
        except Exception:
            if _reduced_dvms_cpp_required():
                raise
            K_elem, raw_rhs = _assemble_kratos_system_core_python(*args)
    else:
        K_elem, raw_rhs = _assemble_kratos_system_core_python(*args)
    return np.asarray(K_elem, dtype=float), np.asarray(raw_rhs, dtype=float)


def assemble_kratos_system_local_blocks_from_field_locals(
    *,
    mesh,
    dh,
    state,
    element_ids: np.ndarray,
    field_locals: Mapping[str, np.ndarray],
    rho_f: float,
    mu_f: float,
    dt: float,
    bossak_alpha: float,
    incompressibility_stabilization_scale: float = 1.0,
    body_force: np.ndarray | None = None,
    use_oss: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the Kratos-matched P1-triangle fluid system from local arrays.

    The returned vector is the runtime local-system raw RHS convention used by
    ``sampled_lspg_rows_from_local_blocks`` after the production driver
    converts ``FluidDVMSLocalVelocityContributionOperator.F_elem`` to raw RHS
    sign.  This function intentionally does not read or write ``Function``
    objects; callers must provide all current and history values on the
    requested element stencil through ``field_locals``.
    """

    eids = np.asarray(element_ids, dtype=int).reshape(-1)
    if eids.size == 0:
        return (
            np.zeros((0, 9, 9), dtype=float),
            np.zeros((0, 9), dtype=float),
            np.zeros((0, 9), dtype=int),
        )
    fast = _dvms_fast_p1_tri_cache(state, dh, mesh)
    if fast is None:
        raise NotImplementedError("array-only Kratos system assembly currently supports only fast P1 triangles.")
    if np.any(eids < 0) or np.any(eids >= int(state.n_elements)):
        raise ValueError("element_ids contain out-of-range entries.")
    if not np.isfinite(float(incompressibility_stabilization_scale)) or float(
        incompressibility_stabilization_scale
    ) <= 0.0:
        raise ValueError("incompressibility_stabilization_scale must be finite and positive.")

    local_inputs = dict(field_locals)
    all_maps = {key: np.asarray(value, dtype=int) for key, value in dict(fast["element_maps"]).items()}
    maps = {key: value[eids] for key, value in all_maps.items()}

    ux = _field_local_array(local_inputs, names=("ux", "ux_k"), field_name="ux", element_maps=maps, all_element_maps=all_maps, element_ids=eids)
    uy = _field_local_array(local_inputs, names=("uy", "uy_k"), field_name="uy", element_maps=maps, all_element_maps=all_maps, element_ids=eids)
    p_vals = _field_local_array(local_inputs, names=("p", "p_k"), field_name="p", element_maps=maps, all_element_maps=all_maps, element_ids=eids)
    mx = _field_local_array(local_inputs, names=("mx", "mx_k"), field_name="mx", element_maps=maps, all_element_maps=all_maps, element_ids=eids)
    my = _field_local_array(local_inputs, names=("my", "my_k"), field_name="my", element_maps=maps, all_element_maps=all_maps, element_ids=eids)
    mx_prev = _field_local_array(local_inputs, names=("mx_prev",), field_name="mx", element_maps=maps, all_element_maps=all_maps, element_ids=eids)
    my_prev = _field_local_array(local_inputs, names=("my_prev",), field_name="my", element_maps=maps, all_element_maps=all_maps, element_ids=eids)
    ax_prev = _field_local_array(local_inputs, names=("ax_prev",), field_name="ux", element_maps=maps, all_element_maps=all_maps, element_ids=eids)
    ay_prev = _field_local_array(local_inputs, names=("ay_prev",), field_name="uy", element_maps=maps, all_element_maps=all_maps, element_ids=eids)

    bossak = _bossak_coefficients(alpha=float(bossak_alpha), dt=max(float(dt), _SINGULAR_TOL))
    if "ax_curr" in local_inputs or "ax" in local_inputs:
        ax_curr = _field_local_array(local_inputs, names=("ax_curr", "ax"), field_name="ux", element_maps=maps, all_element_maps=all_maps, element_ids=eids)
        ay_curr = _field_local_array(local_inputs, names=("ay_curr", "ay"), field_name="uy", element_maps=maps, all_element_maps=all_maps, element_ids=eids)
    else:
        ux_prev = _field_local_array(local_inputs, names=("ux_prev",), field_name="ux", element_maps=maps, all_element_maps=all_maps, element_ids=eids)
        uy_prev = _field_local_array(local_inputs, names=("uy_prev",), field_name="uy", element_maps=maps, all_element_maps=all_maps, element_ids=eids)
        ax_curr = float(bossak["ma0"]) * (ux - ux_prev) + float(bossak["ma2"]) * ax_prev
        ay_curr = float(bossak["ma0"]) * (uy - uy_prev) + float(bossak["ma2"]) * ay_prev

    if "mx_vel" in local_inputs or "my_vel" in local_inputs:
        mx_vel = _field_local_array(local_inputs, names=("mx_vel",), field_name="mx", element_maps=maps, all_element_maps=all_maps, element_ids=eids)
        my_vel = _field_local_array(local_inputs, names=("my_vel",), field_name="my", element_maps=maps, all_element_maps=all_maps, element_ids=eids)
    elif (
        ("mx_vel_prev" in local_inputs or "my_vel_prev" in local_inputs)
        and ("mx_acc_prev" in local_inputs or "my_acc_prev" in local_inputs)
    ):
        mx_vel_prev = _field_local_array(local_inputs, names=("mx_vel_prev",), field_name="mx", element_maps=maps, all_element_maps=all_maps, element_ids=eids)
        my_vel_prev = _field_local_array(local_inputs, names=("my_vel_prev",), field_name="my", element_maps=maps, all_element_maps=all_maps, element_ids=eids)
        mx_acc_prev = _field_local_array(local_inputs, names=("mx_acc_prev",), field_name="mx", element_maps=maps, all_element_maps=all_maps, element_ids=eids)
        my_acc_prev = _field_local_array(local_inputs, names=("my_acc_prev",), field_name="my", element_maps=maps, all_element_maps=all_maps, element_ids=eids)
        beta = float(bossak["beta"])
        gamma = float(bossak["gamma"])
        dt_value = max(float(dt), _SINGULAR_TOL)
        mx_acc = (mx - mx_prev - dt_value * mx_vel_prev - dt_value * dt_value * (0.5 - beta) * mx_acc_prev) / (
            beta * dt_value * dt_value
        )
        my_acc = (my - my_prev - dt_value * my_vel_prev - dt_value * dt_value * (0.5 - beta) * my_acc_prev) / (
            beta * dt_value * dt_value
        )
        mx_vel = mx_vel_prev + dt_value * ((1.0 - gamma) * mx_acc_prev + gamma * mx_acc)
        my_vel = my_vel_prev + dt_value * ((1.0 - gamma) * my_acc_prev + gamma * my_acc)
    elif "mx_prev2" in local_inputs or "my_prev2" in local_inputs:
        mx_prev2 = _field_local_array(local_inputs, names=("mx_prev2",), field_name="mx", element_maps=maps, all_element_maps=all_maps, element_ids=eids)
        my_prev2 = _field_local_array(local_inputs, names=("my_prev2",), field_name="my", element_maps=maps, all_element_maps=all_maps, element_ids=eids)
        dt_value = max(float(dt), _SINGULAR_TOL)
        mx_vel = (1.5 * mx - 2.0 * mx_prev + 0.5 * mx_prev2) / dt_value
        my_vel = (1.5 * my - 2.0 * my_prev + 0.5 * my_prev2) / dt_value
    else:
        dt_value = max(float(dt), _SINGULAR_TOL)
        mx_vel = (mx - mx_prev) / dt_value
        my_vel = (my - my_prev) / dt_value

    rho = float(rho_f)
    mu = float(mu_f)
    dt_value = max(float(dt), _SINGULAR_TOL)
    inv_dt = 1.0 / dt_value
    inc_scale = float(incompressibility_stabilization_scale)
    body = np.zeros(2, dtype=float) if body_force is None else np.asarray(body_force, dtype=float).reshape(2)
    mam = (1.0 - float(bossak["alpha"])) * float(bossak["ma0"])

    n_elem = int(eids.size)
    n_q = int(state.n_qp_per_element)
    basis_u = np.asarray(fast["basis"]["ux"], dtype=float)
    basis_p = np.asarray(fast["basis"]["p"], dtype=float)
    J_inv = np.asarray(fast["J_inv"], dtype=float)[eids]
    detJ_geo = np.asarray(fast["detJ"], dtype=float)[eids]
    ref_weights = np.asarray(state.quadrature_layout.reference_weights, dtype=float).reshape(-1)
    if ref_weights.size != n_q:
        raise RuntimeError("DVMS quadrature weight count does not match state layout.")

    grad_phi_u_ref = _grad_phi_phys(np.asarray(fast["grad_ref"]["ux"], dtype=float), J_inv)
    grad_phi_p_ref = _grad_phi_phys(np.asarray(fast["grad_ref"]["p"], dtype=float), J_inv)
    grad_phi_mx_ref = _grad_phi_phys(np.asarray(fast["grad_ref"]["mx"], dtype=float), J_inv)
    grad_phi_my_ref = _grad_phi_phys(np.asarray(fast["grad_ref"]["my"], dtype=float), J_inv)

    grad_u_ref = np.zeros((n_elem, 2, 2), dtype=float)
    grad_u_ref[:, 0, :] = np.einsum("el,elk->ek", ux, grad_phi_u_ref, optimize=True)
    grad_u_ref[:, 1, :] = np.einsum("el,elk->ek", uy, grad_phi_u_ref, optimize=True)
    grad_d_ref = np.zeros((n_elem, 2, 2), dtype=float)
    grad_d_ref[:, 0, :] = np.einsum("el,elk->ek", mx, grad_phi_mx_ref, optimize=True)
    grad_d_ref[:, 1, :] = np.einsum("el,elk->ek", my, grad_phi_my_ref, optimize=True)
    grad_p_ref = np.einsum("el,elk->ek", p_vals, grad_phi_p_ref, optimize=True)

    F = np.zeros((n_elem, 2, 2), dtype=float)
    F[:, 0, 0] = 1.0 + grad_d_ref[:, 0, 0]
    F[:, 0, 1] = grad_d_ref[:, 0, 1]
    F[:, 1, 0] = grad_d_ref[:, 1, 0]
    F[:, 1, 1] = 1.0 + grad_d_ref[:, 1, 1]
    detF = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
    bad = np.argwhere(~np.isfinite(detF) | (np.abs(detF) <= _SINGULAR_TOL))
    if bad.size:
        raise RuntimeError(f"Singular ALE deformation gradient on element {int(eids[int(bad[0, 0])])}.")
    Finv = np.zeros_like(F)
    Finv[:, 0, 0] = F[:, 1, 1] / detF
    Finv[:, 0, 1] = -F[:, 0, 1] / detF
    Finv[:, 1, 0] = -F[:, 1, 0] / detF
    Finv[:, 1, 1] = F[:, 0, 0] / detF
    cof = np.zeros_like(F)
    cof[:, 0, 0] = F[:, 1, 1]
    cof[:, 0, 1] = -F[:, 1, 0]
    cof[:, 1, 0] = -F[:, 0, 1]
    cof[:, 1, 1] = F[:, 0, 0]

    grad_u_phys = np.einsum("eij,ejk->eik", grad_u_ref, Finv, optimize=True)
    grad_p_phys = np.einsum("eji,ej->ei", Finv, grad_p_ref, optimize=True)
    grad_phi_u_cur = np.einsum("eli,eij->elj", grad_phi_u_ref, Finv, optimize=True)
    grad_phi_p_cur = np.einsum("eli,eij->elj", grad_phi_p_ref, Finv, optimize=True)

    u_q = np.stack(
        [
            np.einsum("el,ql->eq", ux, basis_u, optimize=True),
            np.einsum("el,ql->eq", uy, basis_u, optimize=True),
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
    resolved_conv = u_q - mesh_v_q

    predicted = np.asarray(state.predicted_subscale_velocity, dtype=float).reshape(int(state.n_elements), n_q, 2)[eids]
    old_subscale = np.asarray(state.old_subscale_velocity, dtype=float).reshape(int(state.n_elements), n_q, 2)[eids]
    if bool(use_oss):
        momentum_projection = np.asarray(state.momentum_projection, dtype=float).reshape(int(state.n_elements), n_q, 2)[eids]
        mass_projection = np.asarray(state.mass_projection, dtype=float).reshape(int(state.n_elements), n_q)[eids]
    else:
        momentum_projection = np.zeros((n_elem, n_q, 2), dtype=float)
        mass_projection = np.zeros((n_elem, n_q), dtype=float)
    old_mass_residual = np.asarray(state.old_mass_residual, dtype=float).reshape(int(state.n_elements), n_q)[eids]

    h_e = _current_element_size_from_local_mesh(mesh, element_ids=eids, mx=mx, my=my)
    if h_e.shape != (n_elem,):
        raise RuntimeError("current element-size local array has an unexpected shape.")

    x_local = np.concatenate([ux, uy, p_vals], axis=1)
    scheme_ax = (1.0 - float(bossak["alpha"])) * ax_curr + float(bossak["alpha"]) * ax_prev
    scheme_ay = (1.0 - float(bossak["alpha"])) * ay_curr + float(bossak["alpha"]) * ay_prev
    a_local = np.concatenate([scheme_ax, scheme_ay, np.zeros_like(p_vals)], axis=1)

    K_elem, raw_rhs = _assemble_kratos_system_core(
        basis_u,
        basis_p,
        detJ_geo,
        ref_weights,
        detF,
        cof,
        grad_phi_u_ref,
        grad_phi_u_cur,
        grad_phi_p_cur,
        grad_u_phys,
        resolved_conv,
        predicted,
        old_subscale,
        momentum_projection,
        mass_projection,
        old_mass_residual,
        h_e,
        x_local,
        a_local,
        body,
        rho,
        mu,
        inv_dt,
        inc_scale,
        mam,
    )
    if not np.all(np.isfinite(K_elem)) or not np.all(np.isfinite(raw_rhs)):
        raise RuntimeError("array-only Kratos system assembly produced non-finite local blocks.")
    # The core formula assembles the symbolic local residual sign.  The sampled
    # HROM algebra is wired to the runtime local-operator raw RHS convention,
    # where the production driver uses ``-local.F_elem`` before forming Newton
    # residual rows.
    return K_elem, -raw_rhs, _local_grouped_gdofs_batch(dh, eids)


__all__ = [
    "ReducedDVMSState",
    "assemble_kratos_system_local_blocks_from_field_locals",
    "reduced_dvms_cpp_backend_status",
    "update_predicted_subscale_local",
]
