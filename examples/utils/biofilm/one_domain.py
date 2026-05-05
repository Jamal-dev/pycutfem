"""One-domain diffuse-interface biofilm model (Navier–Stokes/Darcy–Biot + transport).

This module implements the weak residual and a manually coded consistent Jacobian
for the model described in `examples/biofilms/model/model.tex`.

Design goals
------------
* Keep the implementation debuggable by exposing per-block residuals.
* Use a one-step-θ scheme (θ=1: backward Euler, θ=0.5: Crank–Nicolson).
* Be compatible with all pycutfem backends ("python", "jit", "cpp").

Important compiler limitation
-----------------------------
The current `FormCompiler` only reliably supports `grad(...)` / `div(...)` when
applied directly to base Trial/Test/Function objects. Avoid `div(a*v)` and
`div(vS_k)` where `vS_k` is a linear combination. This module expands such
divergences explicitly (product rule / difference-of-divergences).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from pycutfem.ufl.expressions import (
    Constant,
    Derivative,
    FacetNormal,
    HdivFunction,
    HdivTestFunction,
    HdivTrialFunction,
    Identity,
    Laplacian,
    MeshSize,
    avg,
    det,
    div,
    dot,
    grad,
    inner,
    jump,
    outer,
    tanh,
    trace,
)

from pycutfem.ufl.linalg import (
    d_spectral_positive_part_2x2_sym,
    spectral_positive_part_2x2_sym,
    smooth_pos as _smooth_pos_u,
    smooth_pos_derivative as _smooth_pos_u_prime,
)

from ..shared.nonlinear_solid_refmap import (
    deulerian_F,
    deulerian_k_inv,
    dsigma_svk,
    dsigma_svk_miehe_split,
    dsigma_hencky,
    dsigma_hencky_miehe_split,
    dsigma_neo_hookean,
    dsigma_neo_hookean_seboldt,
    eulerian_F,
    eulerian_k_inv,
    svk_tensile_energy_miehe,
    hencky_tensile_energy_miehe,
    sigma_svk,
    sigma_svk_miehe_split,
    sigma_hencky,
    sigma_hencky_miehe_split,
    sigma_neo_hookean,
    sigma_neo_hookean_seboldt,
)


def _c(val: float) -> Constant:
    return Constant(float(val))


def _as_constant(val) -> Constant:
    return val if isinstance(val, Constant) else _c(float(val))


def _mark_runtime_parameter(val, name: str):
    if isinstance(val, Constant):
        setattr(val, "_preserve_runtime_structure", True)
        if not getattr(val, "_jit_name", None):
            setattr(val, "_jit_name", str(name))
    return val


def _as_scalar_expr(val):
    if isinstance(val, Constant):
        return val
    try:
        return _c(float(val))
    except Exception:
        return val


def _sqrt(expr):
    return expr ** _c(0.5)


def _one_minus(expr):
    # IMPORTANT: keep the "function-like" operand on the left so VecOpInfo handles
    # the arithmetic (float - VecOpInfo is not supported by the compiler).
    return (-expr) + _c(1.0)


def _epsilon(v):
    return _c(0.5) * (grad(v) + grad(v).T)


def _is_seboldt_neo_hookean_model(model_key: str) -> bool:
    key = str(model_key).strip().lower().replace("-", "_")
    return key in {"seboldt_neo_hookean", "neo_hookean_seboldt", "seboldt_nh", "nh_seboldt"}


def _num_components(expr, default: int = 2) -> int:
    if hasattr(expr, "num_components"):
        try:
            ncomp = int(getattr(expr, "num_components", 0))
            if ncomp > 0:
                return ncomp
        except Exception:
            pass
    shape = getattr(expr, "shape", None)
    if isinstance(shape, tuple) and len(shape) == 1:
        try:
            ncomp = int(shape[0])
            if ncomp > 0:
                return ncomp
        except Exception:
            pass
    return int(default)


def _is_hdiv_expr(expr) -> bool:
    return isinstance(expr, (HdivFunction, HdivTrialFunction, HdivTestFunction))


def _grad_component(vec_expr, i: int, j: int):
    return grad(_vector_component(vec_expr, i))[j]


def _epsilon_component(v, i: int, j: int):
    if int(i) == int(j):
        return _grad_component(v, i, j)
    return _c(0.5) * (_grad_component(v, i, j) + _grad_component(v, j, i))


def _grad_inner_jump(u, v, n):
    """Penalty on the jump of the normal derivative across an interior facet."""
    # Use a componentwise normal-derivative jump for vector fields. This keeps
    # the symbolic form explicit and avoids tensor/jump contraction ambiguity in
    # mixed backend paths while remaining mathematically identical to
    # inner(jump(grad(u), n), jump(grad(v), n)).
    ncomp = 0
    if hasattr(u, "num_components"):
        try:
            ncomp = int(getattr(u, "num_components", 0))
        except Exception:
            ncomp = 0
    if ncomp <= 0:
        shape_u = getattr(u, "shape", None)
        if isinstance(shape_u, tuple) and len(shape_u) == 1:
            try:
                ncomp = int(shape_u[0])
            except Exception:
                ncomp = 0
    if ncomp <= 1:
        ju = jump(grad(u), n)
        jv = jump(grad(v), n)
        return inner(ju, jv)

    acc = _c(0.0)
    for i in range(ncomp):
        acc += jump(grad(_vector_component(u, i)), n) * jump(grad(_vector_component(v, i)), n)
    return acc


def _linear_elastic_term(u, eta, *, mu_s, lambda_s):
    # Keep the elastic contraction on the explicit component path. This avoids
    # relying on matrix-valued grad/transpose lowering in the assembled
    # benchmark blocks and matches the audited `grad(component)` semantics.
    dim = min(_num_components(u), _num_components(eta))
    sym_inner = _c(0.0)
    for i in range(int(dim)):
        for j in range(int(dim)):
            sym_inner += _epsilon_component(u, i, j) * _epsilon_component(eta, i, j)
    return _c(2.0) * mu_s * sym_inner + lambda_s * div(u) * div(eta)


def _deviatoric_tensor(A, *, dim: int):
    return A - (trace(A) / _c(float(dim))) * Identity(int(dim))


def _linear_deviatoric_elastic_term(u, eta, *, mu_s, dim: int):
    dim_i = int(dim)
    tr_u = _c(0.0)
    tr_eta = _c(0.0)
    for i in range(dim_i):
        tr_u += _epsilon_component(u, i, i)
        tr_eta += _epsilon_component(eta, i, i)
    dev_inner = _c(0.0)
    inv_dim = _c(1.0 / float(dim_i))
    for i in range(dim_i):
        for j in range(dim_i):
            delta_ij = _c(1.0) if i == j else _c(0.0)
            dev_u_ij = _epsilon_component(u, i, j) - (tr_u * inv_dim * delta_ij)
            dev_eta_ij = _epsilon_component(eta, i, j) - (tr_eta * inv_dim * delta_ij)
            dev_inner += dev_u_ij * dev_eta_ij
    return _c(2.0) * mu_s * dev_inner


def _vector_component(vec_expr, idx: int):
    try:
        return vec_expr[idx]
    except Exception:
        try:
            basis = Constant([1.0, 0.0] if int(idx) == 0 else [0.0, 1.0], dim=1)
            return dot(vec_expr, basis)
        except Exception:
            pass
        val = getattr(vec_expr, "value", None)
        if val is None:
            raise
        return _c(float(val[int(idx)]))


def _zero_scalar_like(expr):
    try:
        return _c(0.0) * _vector_component(expr, 0)
    except Exception:
        return _c(0.0)


def _dot_2d_components(vec_expr, vec_test):
    return _vector_component(vec_expr, 0) * _vector_component(vec_test, 0) + _vector_component(vec_expr, 1) * _vector_component(vec_test, 1)


def _dot_components(vec_expr, vec_test, *, dim: int):
    acc = _c(0.0)
    for i in range(int(dim)):
        acc += _vector_component(vec_expr, i) * _vector_component(vec_test, i)
    return acc


def _weighted_dot_components(weight, vec_expr, vec_test, *, dim: int):
    acc = _c(0.0)
    for i in range(int(dim)):
        acc += weight * _vector_component(vec_expr, i) * _vector_component(vec_test, i)
    return acc


def _apply_invperm_components(invperm_expr, vec_expr, *, dim: int):
    if getattr(invperm_expr, "dim", None) == 0:
        return tuple(invperm_expr * _vector_component(vec_expr, i) for i in range(int(dim)))
    acc = []
    for i in range(int(dim)):
        comp = _c(0.0)
        for j in range(int(dim)):
            comp += invperm_expr[i, j] * _vector_component(vec_expr, j)
        acc.append(comp)
    return tuple(acc)


def _apply_perm_from_invperm_components(invperm_expr, vec_expr, *, dim: int, dinvperm_expr=None, dvec_expr=None):
    dim_i = int(dim)
    if getattr(invperm_expr, "dim", None) == 0:
        perm_expr = tuple((_c(1.0) / invperm_expr) * _vector_component(vec_expr, i) for i in range(dim_i))
        if dinvperm_expr is None and dvec_expr is None:
            return perm_expr, None
        dperm_expr = []
        for i in range(dim_i):
            comp = _c(0.0)
            if dvec_expr is not None:
                comp += (_c(1.0) / invperm_expr) * _vector_component(dvec_expr, i)
            if dinvperm_expr is not None:
                comp += (-(dinvperm_expr) / (invperm_expr * invperm_expr)) * _vector_component(vec_expr, i)
            dperm_expr.append(comp)
        return perm_expr, tuple(dperm_expr)
    if dim_i != 2:
        raise NotImplementedError("Primary p_pore Darcy elimination is currently implemented for scalar or 2D matrix permeability only.")
    det_k = invperm_expr[0, 0] * invperm_expr[1, 1] - invperm_expr[0, 1] * invperm_expr[1, 0]
    perm_expr = (
        (invperm_expr[1, 1] * _vector_component(vec_expr, 0) - invperm_expr[0, 1] * _vector_component(vec_expr, 1)) / det_k,
        (-invperm_expr[1, 0] * _vector_component(vec_expr, 0) + invperm_expr[0, 0] * _vector_component(vec_expr, 1)) / det_k,
    )
    if dinvperm_expr is None and dvec_expr is None:
        return perm_expr, None
    dperm_expr = [_c(0.0), _c(0.0)]
    if dvec_expr is not None:
        dperm_expr[0] += (
            invperm_expr[1, 1] * _vector_component(dvec_expr, 0) - invperm_expr[0, 1] * _vector_component(dvec_expr, 1)
        ) / det_k
        dperm_expr[1] += (
            -invperm_expr[1, 0] * _vector_component(dvec_expr, 0) + invperm_expr[0, 0] * _vector_component(dvec_expr, 1)
        ) / det_k
    if dinvperm_expr is not None:
        dcore_rhs = _apply_invperm_components(dinvperm_expr, perm_expr, dim=dim_i)
        dperm_expr[0] += -(invperm_expr[1, 1] * dcore_rhs[0] - invperm_expr[0, 1] * dcore_rhs[1]) / det_k
        dperm_expr[1] += -(-invperm_expr[1, 0] * dcore_rhs[0] + invperm_expr[0, 0] * dcore_rhs[1]) / det_k
    return perm_expr, tuple(dperm_expr)


def _components(vec_expr, *, dim: int):
    return tuple(_vector_component(vec_expr, i) for i in range(int(dim)))


def _tangent_from_normal_2d(n):
    return (_vector_component(n, 1), -_vector_component(n, 0))


def _tangential_component_2d(vec_expr, n):
    t = _tangent_from_normal_2d(n)
    return _vector_component(vec_expr, 0) * t[0] + _vector_component(vec_expr, 1) * t[1]


def _tangential_symgrad_contraction_2d(v_expr, n):
    t = _tangent_from_normal_2d(n)
    acc = _c(0.0)
    for i in range(2):
        for j in range(2):
            acc += _epsilon_component(v_expr, i, j) * t[i] * _vector_component(n, j)
    return acc


def _tangential_viscous_traction_2d(v_expr, mu_expr, n):
    """
    Tangential scalar traction t · (2 mu eps(v) n) on a 2D boundary.

    This is the natural quantity for weakly imposing tangential Dirichlet data
    on H(div) velocity fields: the normal component is enforced strongly via the
    RT trace DOFs, while the tangential slip/no-slip condition is added with a
    Nitsche term in the momentum equation.
    """
    return _c(2.0) * mu_expr * _tangential_symgrad_contraction_2d(v_expr, n)


def _grad_div_components(v_expr, *, dim: int, hdiv_order: int = 0):
    """
    Exact componentwise expansion of grad(div(v)).

    In 2D:
      [grad(div(v))]_0 = d_xx v_0 + d_xy v_1
      [grad(div(v))]_1 = d_xy v_0 + d_yy v_1
    """
    if int(dim) != 2:
        raise NotImplementedError("grad(div(v)) component expansion is currently implemented for 2D only.")
    v0 = _vector_component(v_expr, 0)
    v1 = _vector_component(v_expr, 1)
    return (
        Derivative(v0, 2, 0) + Derivative(v1, 1, 1),
        Derivative(v0, 1, 1) + Derivative(v1, 0, 2),
    )


def _laplace_components(vec_expr, *, dim: int):
    comps = []
    for i in range(int(dim)):
        vi = _vector_component(vec_expr, i)
        comps.append(Derivative(vi, 2, 0) + Derivative(vi, 0, 2))
    return tuple(comps)


def _strong_div_2mu_eps_components(v_expr, mu_expr, grad_mu_components, *, dim: int, hdiv_order: int = 0):
    """
    Componentwise strong viscous operator div(2 mu eps(v)).

    We expand
        div(2 mu eps(v)) = mu (Delta v + grad(div v)) + 2 eps(v) grad(mu)
    componentwise using primitive operators already exercised by the backends.
    This avoids relying on tensor-divergence code paths for composite expressions.
    """
    lap_v = _laplace_components(v_expr, dim=int(dim))
    grad_div_v = _grad_div_components(v_expr, dim=int(dim), hdiv_order=int(hdiv_order))
    comps = []
    for i in range(int(dim)):
        comp = mu_expr * (lap_v[i] + grad_div_v[i])
        grad_part = _c(0.0)
        for j in range(int(dim)):
            grad_part += _c(2.0) * _epsilon_component(v_expr, i, j) * grad_mu_components[j]
        comps.append(comp + grad_part)
    return tuple(comps)


def _d_strong_div_2mu_eps_components(
    v_expr,
    dv_expr,
    mu_expr,
    dmu_expr,
    grad_mu_components,
    grad_dmu_components,
    *,
    dim: int,
    hdiv_order: int = 0,
):
    """
    Componentwise variation of div(2 mu eps(v)):
        d[div(2 mu eps(v))]
        = dmu (Delta v + grad(div v))
          + mu (Delta dv + grad(div dv))
          + 2 eps(dv) grad(mu)
          + 2 eps(v) grad(dmu).
    """
    lap_v = _laplace_components(v_expr, dim=int(dim))
    lap_dv = _laplace_components(dv_expr, dim=int(dim))
    grad_div_v = _grad_div_components(v_expr, dim=int(dim), hdiv_order=int(hdiv_order))
    grad_div_dv = _grad_div_components(dv_expr, dim=int(dim), hdiv_order=int(hdiv_order))
    comps = []
    for i in range(int(dim)):
        comp = dmu_expr * (lap_v[i] + grad_div_v[i])
        comp += mu_expr * (lap_dv[i] + grad_div_dv[i])
        grad_part = _c(0.0)
        for j in range(int(dim)):
            grad_part += _c(2.0) * (
                _epsilon_component(dv_expr, i, j) * grad_mu_components[j]
                + _epsilon_component(v_expr, i, j) * grad_dmu_components[j]
            )
        comps.append(comp + grad_part)
    return tuple(comps)


def _smooth_pos(x, *, eta: float = 1.0e-12):
    """Smooth positive part ⟨x⟩_+ ≈ 0.5 (x + sqrt(x^2 + eta))."""
    return _c(0.5) * (x + _sqrt(x * x + _c(float(eta))))


def _chi_b(alpha):
    return alpha


def _chi_f(alpha):
    return _one_minus(alpha)


def _support_indicator(alpha, *, beta: float = 0.0):
    beta_val = float(beta)
    if beta_val <= 0.0:
        return alpha
    z = _c(beta_val) * (alpha - _c(0.5))
    return _c(0.5) * (_c(1.0) + tanh(z))


def _support_indicator_prime(alpha, *, beta: float = 0.0):
    beta_val = float(beta)
    if beta_val <= 0.0:
        return _c(1.0)
    z = _c(beta_val) * (alpha - _c(0.5))
    th = tanh(z)
    return _c(0.5 * beta_val) * (_c(1.0) - th * th)


def _support_indicator_second(alpha, *, beta: float = 0.0):
    beta_val = float(beta)
    if beta_val <= 0.0:
        return _c(0.0)
    z = _c(beta_val) * (alpha - _c(0.5))
    th = tanh(z)
    sech2 = _c(1.0) - th * th
    return _c(-(beta_val * beta_val)) * th * sech2


def _capacity(alpha, phi):
    # C = (1-α) + α φ
    return _chi_f(alpha) + _chi_b(alpha) * phi


def _C_from_B(B):
    # C = 1 - B
    return _one_minus(B)


def _P_from_alpha_B(alpha, B):
    # P = alpha - B
    return alpha - B


def _rho(alpha, phi, *, rho_f):
    # ρ = (1-α) ρ_f + α ρ_f φ = ρ_f ((1-α) + α φ)
    return rho_f * _capacity(alpha, phi)


def _rho_from_B(B, *, rho_f):
    # ρ = ρ_f C = ρ_f (1-B)
    return rho_f * _C_from_B(B)


def _mu(alpha, phi, *, mu_f, mu_b=None, mu_b_model: str = "phi_mu"):
    """
    Effective viscosity μ(α,φ).

    Choices:
      - "mu":      μ_B = μ_f (constant)               → μ = μ_f (no α/φ dependence)
      - "phi_mu":  μ_B = φ μ_f (Brinkman scaling)     → μ = μ_f ((1-α) + α φ)
      - "alpha_mu": μ_B = μ_b (constant)              → μ = (1-α) μ_f + α μ_b
      - "alpha_phi_mu": μ_B = φ μ_b                   → μ = (1-α) μ_f + α φ μ_b
    """
    mu_b_model = str(mu_b_model).strip().lower()
    if mu_b_model in {"mu", "const", "constant"}:
        mu_b_eff = mu_f
    elif mu_b_model in {"phi_mu", "phi*mu", "phi"}:
        mu_b_eff = phi * mu_f
    elif mu_b_model in {"alpha_mu", "alpha", "mu_b", "biofilm_mu", "biofilm"}:
        if mu_b is None:
            raise ValueError("mu_b_model='alpha_mu' requires mu_b to be provided.")
        mu_b_eff = mu_b
    elif mu_b_model in {"alpha_phi_mu", "alpha_phi", "alpha*phi_mu", "alpha*phi*mu"}:
        if mu_b is None:
            raise ValueError("mu_b_model='alpha_phi_mu' requires mu_b to be provided.")
        mu_b_eff = phi * mu_b
    else:
        raise ValueError(f"Unknown mu_b_model {mu_b_model!r}.")
    return _chi_f(alpha) * mu_f + _chi_b(alpha) * mu_b_eff


def _mu_from_alpha_B(alpha, B, *, mu_f, mu_b=None, mu_b_model: str = "phi_mu"):
    """
    Ratio-free effective viscosity μ(α,B) using P = α - B and C = 1 - B.

    Choices:
      - "mu":            μ = μ_f
      - "phi_mu":        μ = μ_f C = μ_f (1-B)
      - "alpha_mu":      μ = (1-α) μ_f + α μ_b
      - "alpha_phi_mu":  μ = (1-α) μ_f + P μ_b
    """
    mu_b_model = str(mu_b_model).strip().lower()
    C = _C_from_B(B)
    P = _P_from_alpha_B(alpha, B)
    if mu_b_model in {"mu", "const", "constant"}:
        return mu_f
    if mu_b_model in {"phi_mu", "phi*mu", "phi"}:
        return mu_f * C
    if mu_b_model in {"alpha_mu", "alpha", "mu_b", "biofilm_mu", "biofilm"}:
        if mu_b is None:
            raise ValueError("mu_b_model='alpha_mu' requires mu_b to be provided.")
        return _chi_f(alpha) * mu_f + _chi_b(alpha) * mu_b
    if mu_b_model in {"alpha_phi_mu", "alpha_phi", "alpha*phi_mu", "alpha*phi*mu"}:
        if mu_b is None:
            raise ValueError("mu_b_model='alpha_phi_mu' requires mu_b to be provided.")
        return _chi_f(alpha) * mu_f + P * mu_b
    raise ValueError(f"Unknown mu_b_model {mu_b_model!r}.")


def _beta(alpha, phi, *, mu_f, kappa_inv):
    # β = α μ_f φ^2 κ^{-1} (here κ^{-1} can be scalar or tensor; we treat it as scalar for now)
    return _chi_b(alpha) * mu_f * (phi * phi) * kappa_inv


def _monod(S, *, mu_max, K_S):
    # Keep Function-like operand on the left: (S + K_S) not (K_S + S).
    return mu_max * (S / (S + K_S))


def _G(S, phi, *, k_g, mu_max, K_S):
    return k_g * _monod(S, mu_max=mu_max, K_S=K_S) * _one_minus(phi)


def _Pi_over_rho_s(S, phi, alpha, *, mu_max, K_S, k_d):
    # Π_b / ρ_s* (model.tex Monod choice) = (μ_max S/(K+S) - k_d) (1-φ) α
    return (_monod(S, mu_max=mu_max, K_S=K_S) - k_d) * _one_minus(phi) * alpha


def _Pi_over_rho_s_B(S, B, *, mu_max, K_S, k_d):
    # Π_b / ρ_s* = (μ_max S/(K+S) - k_d) B
    return (_monod(S, mu_max=mu_max, K_S=K_S) - k_d) * B


def _R_S_consumption(S, phi, alpha, *, mu_max, K_S, k_d, Y):
    # R_S = (1/Y) Π_b/ρ_s*  (sink in the strong form) where Π_b/ρ_s* has units [1/s].
    #
    # Unit convention used by the default implementation:
    # - `_Pi_over_rho_s` returns Π_b/ρ_s* with units [1/s] if (mu_max,k_d) are rates [1/s].
    # - If the substrate variable S is normalized by ρ_s* (dimensionless), then R_S above
    #   is consistent.
    #
    # If instead S is a *mass concentration* [kg/m^3], then the physical sink is
    #   R_S = Π_b / Y = (ρ_s* / Y) (Π_b/ρ_s*).
    #
    # This scaling is handled in `build_biofilm_one_domain_forms` via the `rho_s_star`
    # parameter.
    return (_c(1.0) / Y) * _Pi_over_rho_s(S, phi, alpha, mu_max=mu_max, K_S=K_S, k_d=k_d)


def _R_S_consumption_B(S, B, *, mu_max, K_S, k_d, Y):
    return (_c(1.0) / Y) * _Pi_over_rho_s_B(S, B, mu_max=mu_max, K_S=K_S, k_d=k_d)


def _detachment_from_shear_prev(
    *,
    v_prev,
    alpha_prev,
    mu_prev,
    k_det,
    eta_n: float = 1.0e-12,
    dim: int = 2,
):
    """
    Build a *lagged* detachment rate D_det from previous-step fields.

    This model uses a *rate coefficient* D_det that is treated as known in the Newton step.
    Detachment is then localized to the diffuse interface via a smooth delta δ(α) (handled
    in the α and X equations).

    Implementation note: while an interface-traction model can be written in terms of
    n = ∇α/||∇α|| and tangential projections, the current pycutfem UFL/compiler stack
    does not robustly support the required tensor products for gradients in all backends.
    We therefore use a simple, robust shear proxy:
        τ ≈ ||ε(v_prev)||_F
        D_det = k_det * sqrt( τ^2 + η )
    """
    tau2 = inner(_epsilon(v_prev), _epsilon(v_prev))
    tau = _sqrt(tau2 + _c(float(eta_n)))
    return k_det * tau


def _support_physics_key(mode: str) -> str:
    key = str(mode).strip().lower().replace("-", "_")
    if key in {"legacy", "legacy_exchange", "legacy_support_exchange", "erosion_to_x", "erosion", "detached_x"}:
        return "legacy_exchange"
    if key in {"internal_conversion", "conserved_support", "support_preserving", "support_preserved"}:
        return "internal_conversion"
    if key in {"stored_support", "stored", "compressible_support", "full_stored", "compressible"}:
        return "stored_support"
    raise ValueError(
        f"Unknown support_physics={mode!r}. Use 'legacy_exchange', 'internal_conversion', or 'stored_support'."
    )


def _split_pore_flux_model_key(mode: str | None) -> str:
    key = str(mode or "direct_darcy").strip().lower().replace("-", "_")
    if key in {"direct", "direct_darcy", "darcy", "grad_p", "darcy_grad_p", "pressure_diffusion"}:
        return "direct_darcy"
    if key in {
        "exact_conservative_p",
        "exact_conservative",
        "conservative_p",
        "p_conservative",
        "p_balance",
        "exact_p",
    }:
        return "exact_conservative_p"
    if key in {
        "exact_total_continuity",
        "total_continuity",
        "total_continuity_p",
        "band_mass_darcy",
        "stoter_band_darcy",
    }:
        return "exact_total_continuity"
    if key in {"reduced", "reduced_divq", "divq", "legacy", "derived_flux"}:
        return "reduced_divq"
    raise ValueError(
        f"Unknown split_pore_flux_model={mode!r}. Use 'direct_darcy', 'exact_conservative_p', 'exact_total_continuity', or 'reduced_divq'."
    )


def _split_pore_momentum_model_key(mode: str | None) -> str:
    key = str(mode or "weighted_divp").strip().lower().replace("-", "_")
    if key in {"weighted_divp", "weighted_div_p", "divp", "legacy", "weighted_p"}:
        return "weighted_divp"
    if key in {"band_alpha", "interface_band", "band_only", "band_p"}:
        return "band_alpha"
    raise ValueError(
        f"Unknown split_pore_momentum_model={mode!r}. Use 'weighted_divP' or 'band_alpha'."
    )


@dataclass(frozen=True)
class BiofilmOneDomainForms:
    jacobian_form: object
    residual_form: object
    r_momentum: object
    r_mass: object
    # Solid kinematics (Eulerian reference-map constraint linking u and vS)
    r_kinematics: object
    r_skeleton: object
    r_phi: object
    r_alpha: object
    # Optional: Cahn–Hilliard chemical potential equation for α (μ_α)
    r_mu_alpha: object | None
    r_damage: object | None
    r_substrate: object
    r_phi_transport: object | None = None
    r_alpha_transport: object | None = None
    r_B: object | None = None
    r_pore: object | None = None
    r_total_mass: object | None = None
    r_momentum_terms: dict[str, object] | None = None
    r_mass_terms: dict[str, object] | None = None
    r_kinematics_terms: dict[str, object] | None = None
    r_skeleton_terms: dict[str, object] | None = None
    # Optional per-block Jacobian contributions (useful for debugging/verification)
    a_momentum: object | None = None
    a_momentum_terms: dict[str, object] | None = None
    a_mass: object | None = None
    a_mass_terms: dict[str, object] | None = None
    a_pore: object | None = None
    a_total_mass: object | None = None
    a_kinematics: object | None = None
    a_skeleton: object | None = None
    a_skeleton_terms: dict[str, object] | None = None
    a_phi: object | None = None
    a_B: object | None = None
    a_alpha: object | None = None
    a_phi_transport: object | None = None
    a_alpha_transport: object | None = None
    a_mu_alpha: object | None = None
    a_damage: object | None = None
    a_substrate: object | None = None
    r_detached: object | None = None
    a_detached: object | None = None
    # Optional: conservative Allen–Cahn (global Lagrange multiplier λ_α)
    r_alpha_lambda: object | None = None
    a_alpha_lambda: object | None = None
    r_drag_lambda: object | None = None
    a_drag_lambda: object | None = None
    r_skeleton_pressure: object | None = None
    a_skeleton_pressure: object | None = None
    r_volumetric: object | None = None
    a_volumetric: object | None = None
    r_domain_lm: object | None = None
    a_domain_lm: object | None = None
    r_domain_lm_terms: dict[str, object] | None = None
    a_domain_lm_terms: dict[str, object] | None = None


def build_biofilm_one_domain_forms(
    *,
    # unknowns at t_{n+1}
    v_k,
    p_k,
    # solid velocity (primary unknown)
    vS_k,
    # Eulerian reference-map variable / skeleton displacement-like field
    u_k,
    alpha_k,
    S_k,
    phi_k=None,
    B_k=None,
    mu_alpha_k=None,
    lambda_drag_k=None,
    # optional: bulk damage / cohesion loss (0=intact, 1=failed)
    d_k=None,
    # optional: detached-biomass concentration at t_{n+1}
    X_k=None,
    # unknowns at t_n
    v_n,
    p_n,
    vS_n,
    u_n,
    alpha_n,
    S_n,
    phi_n=None,
    B_n=None,
    mu_alpha_n=None,
    lambda_drag_n=None,
    # optional: bulk damage at t_n
    d_n=None,
    # optional: detached-biomass concentration at t_n
    X_n=None,
    # Newton increments
    dv,
    dp,
    dvS,
    du,
    dalpha,
    dS,
    dphi=None,
    dB=None,
    dmu_alpha=None,
    dd=None,
    dX=None,
    dlambda_drag=None,
    # test functions
    v_test,
    q_test,
    vS_test,
    u_test,
    alpha_test,
    S_test,
    phi_test=None,
    B_test=None,
    mu_alpha_test=None,
    d_test=None,
    X_test=None,
    lambda_drag_test=None,
    # measure
    dx,
    # time integration
    dt,
    p_pore_k=None,
    p_pore_n=None,
    dp_pore=None,
    q_pore_test=None,
    theta: float = 1.0,
    # physical parameters
    rho_f=None,
    mu_f=None,
    mu_b=None,
    kappa_inv=None,
    mu_s=None,
    lambda_s=None,
    # Optional Kelvin–Voigt viscoelasticity for the skeleton (small-strain only):
    # σ_visc = 2 η_s ε(v^S), with v^S treated as a primary unknown.
    solid_visco_eta: float = 0.0,
    # optional solid inertia (Eulerian skeleton acceleration)
    rho_s0_tilde=None,
    include_skeleton_acceleration: bool = False,
    # Optional runtime coefficient for the skeleton inertia block. When
    # provided, the inertia terms are assembled once and scaled with this
    # coefficient rather than being structurally gated by
    # `include_skeleton_acceleration`.
    skeleton_acceleration_weight=None,
    # How to treat the convective part of the Eulerian skeleton inertia
    # div(ρ_S v^S ⊗ v^S). "lagged" (default) uses a Picard-like linearization
    # div(ρ_S^n v^{S,n} ⊗ v^{S,k}) for robustness; "full" keeps the nonlinear
    # term.
    skeleton_inertia_convection: str = "lagged",
    # Optional parametric blend weights for the skeleton inertia-convection
    # variants. When any of these are provided, the builder assembles the
    # requested variants once and scales them with the supplied runtime
    # coefficients instead of branching structurally on
    # `skeleton_inertia_convection`.
    skeleton_inertia_full_weight=None,
    skeleton_inertia_lagged_weight=None,
    # How to treat the convective part of the fluid momentum.
    #
    # The conservative convection in this one-domain formulation includes both
    #   ρ (v·∇)v  and  v div(ρ v),
    # where ρ = ρ_f C(α,φ). This can be a major source of nonlinearity in the
    # monolithic Newton solve.
    #
    # - "full" (default): fully nonlinear convection at the k-level
    # - "lagged": Picard/IMEX-like linearization using n-level coefficients/advectors
    # - "off": omit convection entirely (Stokes/Brinkman limit)
    fluid_convection: str = "full",
    # Optional parametric blend weights for the fluid-convection variants.
    # When any of these are provided, the builder assembles the corresponding
    # terms once and scales them with the supplied runtime coefficients instead
    # of branching structurally on `fluid_convection`.
    fluid_convection_full_weight=None,
    fluid_convection_lagged_weight=None,
    fluid_convection_imex_weight=None,
    # solid/permeability modelling toggles
    solid_model: str = "linear",
    c_nh=None,
    beta_nh=None,
    kappa_inv_model: str = "spatial",
    kappa_inv_phi_ref: float = 0.3,
    kappa_inv_kc_eps: float = 1.0e-12,
    # transport parameters
    D_phi: float = 0.0,
    phi_diffusion_weight: str = "unity",
    gamma_phi: float = 0.0,
    # transport stabilizations (consistent, for advection-dominated cases)
    phi_supg: float = 0.0,
    phi_cip: float = 0.0,
    # Optional storage coefficient for the split-pressure pore row.
    # On the split-pressure stored-support branch the pore equation can be
    # assembled in two ways:
    #
    # - "direct_darcy":  beta * div(vS) - div(alpha K/mu grad(p_P)) + beta c0 D_s p_P = 0
    # - "exact_conservative_p":
    #                    d_t P + div(P vS - alpha K/mu grad(p_P)) + beta c0 D_s p_P = alpha s_v
    #                    with P = alpha - B
    # - "reduced_divq":  beta * div(vS) + div(P (v-vS))            + beta c0 D_s p_P = 0
    #
    # The storage term is localized by the same support coefficient used by the
    # selected pressure-coupling mode, and it is zero by default so the legacy
    # incompressible behavior is unchanged.
    storativity_c0: float = 0.0,
    primary_darcy_flux: bool = False,
    split_pore_flux_model: str = "direct_darcy",
    split_pore_momentum_model: str = "weighted_divP",
    fluid_mass_model: str = "transported_free_content",
    gamma_u: float = 0.0,
    u_extension_mode: str = "l2",
    gamma_u_pin: float = 0.0,
    regularization_weight=None,
    # Optional SUPG-like streamline diffusion stabilizations:
    # - v_supg: fluid momentum convection stabilization
    # - u_supg: kinematic (u) transport stabilization
    v_supg: float = 0.0,
    v_supg_mode: str = "streamline",
    v_supg_c_nu: float = 4.0,
    fluid_hdiv_order: int = 0,
    u_supg: float = 0.0,
    # Optional CIP (interior penalty) stabilizations on the mesh skeleton:
    # - v_cip:  fluid velocity regularization (helps at moderate/high Re on coarse meshes)
    # - vS_cip: skeleton velocity regularization (helps near the diffuse interface)
    v_cip: float = 0.0,
    vS_cip: float = 0.0,
    # Optional augmented-Lagrangian / grad-div style stabilization:
    # adds γ_div * (div(C v + B vS), div(C w) + div(B η)) to the momentum/skeleton
    # equations. This is consistent (vanishes when the constraint holds) and can
    # improve conditioning of the transient convection-dominated solve.
    gamma_div: float = 0.0,
    drag_formulation: str = "direct",
    D_alpha: float = 0.0,
    # Which velocity advects the diffuse indicator α:
    # - "vS"       (legacy default): skeleton/interface velocity v^S
    # - "v":                  fluid velocity v
    # - "biofilm_volume":     occupied-biofilm volume velocity φ v + (1-φ) v^S
    # - "relative":           relative/slip velocity v - v^S
    # - "mix":                mixture/volume velocity F = C v + B v^S, with C=(1-α)+αφ and B=α(1-φ)
    # - "mix_biofilm":        like "mix", but gate the fluid part (Cv) by a smooth α-cutoff to
    #                         avoid advecting α through the pure-fluid (α≈0) region
    #
    # Legacy default: the historical one-domain alpha law used the full mixture
    # flux `mix`, which contains the pure-fluid contribution (1-alpha) v.
    #
    # For the support-preserving formulation selected with
    # `support_physics="internal_conversion"`, the physically consistent choice
    # is `alpha_advect_with="biofilm_volume"` together with
    # `alpha_advection_form="conservative_weak"`.
    #
    # When omitted/None, the builder selects the support-aware default:
    # `biofilm_volume` for `internal_conversion`, `mix` for `legacy_exchange`.
    alpha_advect_with: str | None = None,
    # Parameters used by alpha_advect_with="mix_biofilm":
    # gate C by  g(α) = α^m / (α^m + α0^m). For α≫α0, g≈1; for α≪α0, g≈0.
    alpha_mix_gate_alpha0: float = 0.1,
    alpha_mix_gate_power: int = 4,
    # Optional lagged support-side gate for the stored-support geometric-alpha
    # transport and the matching kinematic displacement transport. When alpha is
    # advected by vS on the stored-support branch, use
    #   u_alpha = g(α^n) vS,   g(α)=α^m/(α^m+α0^m),
    # and use the same lagged gate inside the kinematic u-equation so the
    # fluid-side extension field does not drive geometric/support transport
    # where α≈0. Disabled when alpha_vS_gate_alpha0 <= 0.
    alpha_vS_gate_alpha0: float = 0.0,
    alpha_vS_gate_power: int = 8,
    # Optional smooth Heaviside reconstruction used by the Seboldt staggered
    # branch so alpha acts as a geometry carrier instead of a diffuse
    # weakening factor in selected support/interface terms.
    support_indicator_beta: float = 0.0,
    # Compatibility placeholder for the benchmark-local driver. The legacy
    # one-domain builder still uses only the beta-smoothed/raw indicator path.
    support_indicator_mode: str = "raw",
    # How to advect α by the chosen velocity field:
    # - "advective":                  u·∇α   (indicator/level-set style)
    # - "conservative":               div(α u) = u·∇α + α div(u)
    # - "conservative_weak":          weak conservative form used for exact
    #                                 alpha-balance tracking up to boundary flux
    alpha_advection_form: str = "conservative_weak",
    # Biofilm-support physics:
    # - "legacy_exchange": historical formulation where alpha can grow/shrink and
    #   interfacial detachment feeds suspended biomass X.
    # - "internal_conversion": conserved-support formulation where alpha is
    #   transported by the biofilm-support flux, growth converts pore fluid to
    #   solid via B=alpha(1-phi), and sloughing is mechanical rather than an
    #   alpha/X source term.
    support_physics: str = "legacy_exchange",
    # Stored-support support-content closure:
    # - "evolve_B":    default current-frame solid-content balance
    # - "freeze_B":    diagnostic mode keeping B fixed in time
    # - "frozen_phi_b": diagnostic mode enforcing B=(1-phi_b) alpha pointwise
    stored_support_content_mode: str = "evolve_B",
    # Reference porosity used by stored_support_content_mode="frozen_phi_b".
    phi_b: float = 0.18,
    # Optional interface-maintenance regularization for α:
    # - "none":      no extra geometric regularization beyond the selected phase-field model
    # - "olsson_nt": conservative flux with normal compression and projected normal/tangential smoothing
    alpha_interface_reg: str = "none",
    alpha_interface_reg_gamma: float = 0.0,
    alpha_interface_reg_eps_normal: float = 1.0,
    alpha_interface_reg_eps_tangent: float = 0.0,
    alpha_interface_reg_eta: float = 1.0e-12,
    # If no α-μ phase-field model is active but μ_α is still present in the mixed space,
    # add a harmless L2 pin so the auxiliary field stays well-posed.
    alpha_mu_aux_pin: float = 1.0,
    # Optional interface traction benchmark hook.
    dGamma=None,
    g_t_k=None,
    g_t_n=None,
    traction_weight_k=None,
    traction_weight_n=None,
    ds_hdiv_tangential=None,
    hdiv_tangential_g_k=None,
    hdiv_tangential_g_n=None,
    hdiv_tangential_gamma: float = 0.0,
    hdiv_tangential_method: str = "penalty",
    # Allen–Cahn / phase-field interface regularization for α
    alpha_cahn_M: float = 0.0,
    alpha_cahn_gamma: float = 0.0,
    alpha_cahn_eps: float = 1.0,
    # Conservative Allen–Cahn: mass-conserving via a global Lagrange multiplier λ_α
    alpha_cahn_conservative: bool = False,
    # Conservative Allen–Cahn implementation:
    # - "unknown" (default): include a global Lagrange multiplier λ_α as an unknown
    #   with the constraint equation ∫ M(α)(μ_α-λ_α)=0.
    # - "eliminate": treat λ_α as a dependent coefficient (computed externally,
    #   e.g. by projecting λ_α = (∫ M μ)/(∫ M)) to avoid ill-conditioning when
    #   using degenerate mobility.
    alpha_cahn_conservative_mode: str = "unknown",
    alpha_cahn_mobility: str = "constant",
    # Optional mobility floor for degenerate Allen–Cahn: M(α)=M0(α(1-α)+m_floor)
    # This can prevent complete "deactivation" of the phase-field regularization in
    # bulk regions when α drifts away from {0,1} due to numerical diffusion.
    alpha_cahn_mobility_floor: float = 0.0,
    lambda_alpha_k=None,
    lambda_alpha_n=None,
    dlambda_alpha=None,
    lambda_alpha_test=None,
    # Optional scaling for the conservative Allen–Cahn constraint equation
    # (improves conditioning when using degenerate mobility so ∫M is small).
    alpha_cahn_lambda_scale=None,
    # Cahn–Hilliard regularization for α (mass-conserving phase-field)
    alpha_ch_M: float = 0.0,
    alpha_ch_gamma: float = 0.0,
    alpha_ch_eps: float = 1.0,
    alpha_ch_mobility: str = "constant",
    # Crack propagation: additional surface speed term V_crack via δ(α)
    alpha_crack_k: float = 0.0,
    alpha_crack_Dc: float = 0.0,
    alpha_crack_m: float = 1.0,
    alpha_crack_gamma_kappa: float = 0.0,
    alpha_crack_eta_kappa: float = 1.0e-12,
    alpha_crack_eta_pos: float = 1.0e-12,
    alpha_crack_eta_mech: float = 1.0e-12,
    alpha_crack_driver: str = "shear",
    V_crack_prev=None,
    # transport stabilizations (consistent, for advection-dominated cases)
    alpha_supg: float = 0.0,
    alpha_cip: float = 0.0,
    u_cip: float = 0.0,
    u_cip_weight: str = "fluid",
    ds_cip=None,
    # Optional open-boundary transport measures. When provided, the weak
    # conservative alpha/B balances retain the corresponding boundary fluxes
    # instead of imposing the default natural no-flux closure.
    ds_alpha_transport=None,
    ds_B_transport=None,
    # Skeleton pressure coupling mode:
    # - "whole_domain" keeps the diffuse one-domain split and uses the same
    #   support coefficient in the reduced pore-storage row and skeleton block
    # - "seboldt" uses the interface-consistent Biot term
    #   -(alpha_biot * p, div(alpha * eta))
    skeleton_pressure_mode: str = "whole_domain",
    # Optional Biot-Willis coefficient override for the split-pressure pore /
    # skeleton coupling.
    alpha_biot: float | None = None,
    # Damage (bulk cohesion loss) model parameters
    damage_k: float = 0.0,
    damage_sigma_cr: float = 0.0,
    damage_m: float = 1.0,
    damage_D: float = 0.0,
    damage_gamma_out: float = 0.0,
    damage_eta_pos: float = 1.0e-12,
    damage_kappa_stiff: float = 1.0e-8,
    damage_kappa_perm: float = 1.0e-8,
    damage_model: str = "kinetic",
    damage_eta: float = 0.0,
    damage_Gc: float = 0.0,
    damage_l: float = 0.0,
    damage_psi0: float = 0.0,
    damage_pf_driver: str = "von_mises",
    # Optional (lagged) phase-field damage history field H^{prev} (prevents healing when the drive relaxes).
    damage_H_prev=None,
    damage_stiff_split: str = "full",
    D_S: float = 0.0,
    # Substrate reaction time discretization:
    # - "theta" (default): use the global θ-scheme (may be CN if θ=0.5)
    # - "implicit"/"imex": treat the reaction term fully implicitly (L-stable for stiff decay)
    # - "explicit": treat the reaction term explicitly (not recommended for stiff kinetics)
    substrate_reaction_scheme: str = "theta",
    # Substrate diffusion time discretization (same choices as reaction scheme).
    substrate_diffusion_scheme: str = "theta",
    D_X: float = 0.0,
    # growth / detachment parameters
    mu_max: float = 0.0,
    K_S: float = 1.0,
    k_g: float = 0.0,
    k_d: float = 0.0,
    Y: float = 1.0,
    rho_s_star: float = 1.0,
    k_det: float = 0.0,
    # modelling toggles
    mu_b_model: str = "phi_mu",
    dim: int = 2,
    # sources (all optional; default to 0)
    f_v=None,
    f_u=None,
    s_v=None,
    ds_v=None,
    f_phi=None,
    f_alpha=None,
    f_S=None,
    f_X=None,
    # detachment rate override (if given, used instead of shear-based lagged rate)
    D_det_prev=None,
    # optional wall adhesion traction (applied to skeleton on a boundary segment)
    ds_adh=None,
    adhesion_k_n: float = 0.0,
    adhesion_k_t: float = 0.0,
    adhesion_gamma_n: float = 0.0,
    adhesion_gamma_t: float = 0.0,
    adhesion_a_prev=None,
    # Solid velocity extension stabilization outside biofilm (α≈0):
    # Keep vS well-posed in a one-domain CG setting by adding a weak extension
    # penalty in the fluid region. By default, we mirror the u-extension settings.
    gamma_vS: float | None = None,
    vS_extension_mode: str | None = None,
    gamma_vS_pin: float | None = None,
    gamma_vS_pin_power: int = 2,
    # Scaling for the kinematic constraint (improves monolithic conditioning).
    kinematics_scale: float | None = None,
    # Optional mixed volumetric-stress reformulation for the linear solid model.
    pi_s_k=None,
    pi_s_n=None,
    dpi_s=None,
    pi_s_test=None,
    solid_volumetric_split: bool = False,
    solid_volumetric_penalty: float = 1.0,
    # Optional consistent lift of the mixed volumetric relation into the mass
    # row. This is an invertible block rotation of the (mass, pi_s) subsystem:
    # the volumetric equation is kept unchanged and an explicitly scaled copy of
    # it is added to the q-test row, so the solution set stays the same while
    # the pressure block is no longer blind to pi_s.
    pressure_block_lift_scale: float = 0.0,
    # Optional formulation-level nondimensionalization factors for the two
    # momentum balances. Multiplying a whole residual equation by a positive
    # scalar leaves the solution set unchanged, but it can remove large
    # constitutive-parameter factors from the assembled Jacobian.
    fluid_momentum_scale: float | None = None,
    skeleton_momentum_scale: float | None = None,
) -> BiofilmOneDomainForms:
    """
    Build residual + consistent Jacobian for the one-domain biofilm system.

    Notes
    -----
    - `dt` may be a float or a `Constant`.
    - All source terms are *added* on the RHS of the strong form; i.e. we build
      residuals of the form "LHS - RHS" so the default is homogeneous (0).
    """
    use_B_primary = any(val is not None for val in (B_k, B_n, dB, B_test))
    if use_B_primary and not all(val is not None for val in (B_k, B_n, dB, B_test)):
        raise ValueError("ratio-free full-state mode requires (B_k, B_n, dB, B_test) together.")
    if use_B_primary and any(val is not None for val in (phi_k, phi_n, dphi, phi_test)):
        raise ValueError("ratio-free full-state mode cannot be mixed with phi as a primary field.")
    if (not use_B_primary) and any(val is None for val in (phi_k, phi_n, dphi, phi_test)):
        raise ValueError("legacy full-state mode requires (phi_k, phi_n, dphi, phi_test).")

    if rho_f is None or mu_f is None or kappa_inv is None:
        raise ValueError("Missing required physical parameters: rho_f, mu_f, kappa_inv.")

    dt = _mark_runtime_parameter(_as_constant(dt), "one_domain_dt")
    rho_f = _mark_runtime_parameter(_as_constant(rho_f), "one_domain_rho_f")
    mu_f = _mark_runtime_parameter(_as_constant(mu_f), "one_domain_mu_f")
    kappa_inv = _mark_runtime_parameter(_as_constant(kappa_inv), "one_domain_kappa_inv")
    if mu_s is not None:
        mu_s = _mark_runtime_parameter(_as_constant(mu_s), "one_domain_mu_s")
    if lambda_s is not None:
        lambda_s = _mark_runtime_parameter(_as_constant(lambda_s), "one_domain_lambda_s")
    if c_nh is not None:
        c_nh = _mark_runtime_parameter(_as_constant(c_nh), "one_domain_c_nh")
    if beta_nh is not None:
        beta_nh = _mark_runtime_parameter(_as_constant(beta_nh), "one_domain_beta_nh")

    solid_model_key = str(solid_model).strip().lower()
    if solid_model_key in {"linear", "small_strain", "linear_elastic"}:
        if mu_s is None or lambda_s is None:
            raise ValueError("Solid model 'linear' requires mu_s and lambda_s.")
    elif solid_model_key in {"hencky", "hencky_log", "hencky_log_strain"}:
        if mu_s is None or lambda_s is None:
            raise ValueError("Solid model 'hencky' requires mu_s and lambda_s.")
    elif solid_model_key in {"stvk", "svk", "saint_venant_kirchhoff", "saint-venant-kirchhoff"}:
        if mu_s is None or lambda_s is None:
            raise ValueError("Solid model 'stvk' requires mu_s and lambda_s.")
    elif solid_model_key in {"neo_hookean", "neo_hookean_eulerian", "nh"}:
        # Allow explicit neo-Hookean parameters; otherwise derive from (mu_s, lambda_s).
        if (c_nh is None or beta_nh is None) and (mu_s is None or lambda_s is None):
            raise ValueError("Solid model 'neo_hookean' requires either (c_nh,beta_nh) or (mu_s,lambda_s).")
    elif _is_seboldt_neo_hookean_model(solid_model_key):
        if mu_s is None or lambda_s is None:
            raise ValueError("Solid model 'seboldt_neo_hookean' requires mu_s and lambda_s.")
    else:
        raise ValueError(f"Unknown solid_model {solid_model!r}.")
    total_pressure_ref = None
    if bool(solid_volumetric_split):
        if solid_model_key not in {"linear", "small_strain", "linear_elastic"} and not _is_seboldt_neo_hookean_model(
            solid_model_key
        ):
            raise ValueError(
                "solid_volumetric_split is currently implemented for the linear and Seboldt Neo-Hookean solid models."
            )
        if pi_s_k is None or pi_s_n is None or dpi_s is None or pi_s_test is None:
            raise ValueError(
                "solid_volumetric_split requires (pi_s_k, pi_s_n, dpi_s, pi_s_test)."
            )
        if skeleton_momentum_scale is not None and abs(float(skeleton_momentum_scale)) > 0.0:
            total_pressure_ref = 1.0 / float(skeleton_momentum_scale)
        else:
            if _is_seboldt_neo_hookean_model(solid_model_key):
                # On the Seboldt Neo-Hookean branch lambda_s matches the paper's
                # volumetric coefficient (2 D_1 = lambda_s), so pi_s should be
                # normalized with that same scale.
                total_pressure_ref = float(lambda_s)
            else:
                total_pressure_ref = 2.0 * float(mu_s) + float(dim) * float(lambda_s)
        if not np.isfinite(total_pressure_ref) or total_pressure_ref <= 0.0:
            raise ValueError("solid_volumetric_split requires a positive total-pressure reference scale.")
    support_physics_key = _support_physics_key(support_physics)
    stored_support_content_key = str(stored_support_content_mode or "evolve_B").strip().lower().replace("-", "_")
    if stored_support_content_key not in {"evolve_b", "freeze_b", "frozen_phi_b"}:
        raise ValueError(
            f"Unknown stored_support_content_mode={stored_support_content_mode!r}. "
            "Use 'evolve_B', 'freeze_B', or 'frozen_phi_b'."
        )
    skel_press_key = str(skeleton_pressure_mode).strip().lower().replace("-", "_")
    if skel_press_key not in {"whole_domain", "seboldt"}:
        raise ValueError(
            f"Unsupported skeleton_pressure_mode={skeleton_pressure_mode!r}. "
            "Use 'whole_domain' or 'seboldt'."
        )
    has_any_pore_pressure_field = any(val is not None for val in (p_pore_k, p_pore_n, dp_pore, q_pore_test))
    has_all_pore_pressure_fields = all(val is not None for val in (p_pore_k, p_pore_n, dp_pore, q_pore_test))
    if bool(has_any_pore_pressure_field) and not bool(has_all_pore_pressure_fields):
        raise ValueError(
            "Split pore-pressure mode requires (p_pore_k, p_pore_n, dp_pore, q_pore_test) together."
        )
    use_split_pore_pressure = bool(support_physics_key == "stored_support" and use_B_primary and has_all_pore_pressure_fields)
    use_primary_darcy_flux = bool(primary_darcy_flux)
    use_single_pressure_primary_darcy_flux = bool(use_primary_darcy_flux and not use_split_pore_pressure)
    split_pore_flux_model_key = _split_pore_flux_model_key(split_pore_flux_model)
    split_pore_momentum_model_key = _split_pore_momentum_model_key(split_pore_momentum_model)
    use_split_pore_direct_darcy = bool(use_split_pore_pressure and (not use_primary_darcy_flux) and split_pore_flux_model_key == "direct_darcy")
    use_split_pore_exact_conservative_p = bool(
        use_split_pore_pressure and (not use_primary_darcy_flux) and split_pore_flux_model_key == "exact_conservative_p"
    )
    use_split_pore_exact_conservative_q = bool(
        use_split_pore_pressure and bool(use_primary_darcy_flux) and split_pore_flux_model_key == "exact_conservative_p"
    )
    use_split_pore_exact_total_continuity = bool(
        use_split_pore_pressure and (not use_primary_darcy_flux) and split_pore_flux_model_key == "exact_total_continuity"
    )
    use_split_pore_exact_total_continuity_q = bool(
        use_split_pore_pressure and bool(use_primary_darcy_flux) and split_pore_flux_model_key == "exact_total_continuity"
    )
    fluid_mass_model_key = str(fluid_mass_model or "transported_free_content").strip().lower().replace("-", "_")
    if fluid_mass_model_key in {"transported", "transported_content", "free_content", "transported_free_content"}:
        fluid_mass_model_key = "transported_free_content"
    elif fluid_mass_model_key in {"fixed_indicator", "rigid_geometry", "fixed_indicator_incompressibility"}:
        fluid_mass_model_key = "fixed_indicator_incompressibility"
    else:
        raise ValueError(
            f"Unknown fluid_mass_model={fluid_mass_model!r}. Use 'transported_free_content' or "
            "'fixed_indicator_incompressibility'."
        )
    if bool(use_split_pore_pressure) and any(val is None for val in (p_pore_k, p_pore_n, dp_pore, q_pore_test)):
        raise ValueError(
            "support_physics='stored_support' requires (p_pore_k, p_pore_n, dp_pore, q_pore_test) "
            "so the pore-pressure storage law is kept separate from the free-fluid pressure block."
        )
    if bool(support_physics_key == "stored_support" and use_B_primary and not use_split_pore_pressure):
        if float(storativity_c0) != 0.0:
            raise ValueError(
                "The single-pressure stored_support(alpha,B,p) core is incompressible and therefore requires "
                "storativity_c0=0. Use the split-pressure branch if you want the reduced Biot pore-storage law."
            )
        if skel_press_key != "whole_domain":
            raise ValueError(
                "The single-pressure stored_support(alpha,B,p) core requires skeleton_pressure_mode='whole_domain' "
                "because p is the multiplier of div(C v + B vS)."
            )
        if alpha_biot is not None:
            raise ValueError(
                "alpha_biot is only part of the split-pressure pore/skeleton Biot coupling. "
                "Do not use alpha_biot on the single-pressure stored_support(alpha,B,p) core."
            )
    drag_form_key = str(drag_formulation or "direct").strip().lower().replace("-", "_")
    if drag_form_key in {"mixed", "mixed_formulation", "mixed_auxiliary", "mixed_lagrange_multiplier"}:
        drag_form_key = "mixed_lm"
    if drag_form_key not in {"direct", "mixed_lm"}:
        raise ValueError(
            f"Unknown drag_formulation={drag_formulation!r}. Use 'direct' or 'mixed_lm'."
        )
    if drag_form_key == "mixed_lm" and (
        lambda_drag_k is None or lambda_drag_n is None or dlambda_drag is None or lambda_drag_test is None
    ):
        raise ValueError(
            "drag_formulation='mixed_lm' requires (lambda_drag_k, lambda_drag_n, dlambda_drag, lambda_drag_test)."
        )
    if bool(use_primary_darcy_flux):
        if drag_form_key != "mixed_lm":
            raise ValueError("primary_darcy_flux=True requires drag_formulation='mixed_lm'.")
        try:
            s_v_is_zero = float(s_v if s_v is not None else _c(0.0)) == 0.0
        except Exception:
            s_v_is_zero = False
        try:
            ds_v_is_zero = float(ds_v if ds_v is not None else _c(0.0)) == 0.0
        except Exception:
            ds_v_is_zero = False
        if not bool(s_v_is_zero and ds_v_is_zero):
            raise ValueError(
                "primary_darcy_flux=True currently supports only the zero-source pore/solid mass law. "
                "Map the occupied-volume source terms explicitly before enabling nonzero s_v/ds_v on this branch."
            )
    q_flux_k = lambda_drag_k if bool(use_primary_darcy_flux) else None
    q_flux_n = lambda_drag_n if bool(use_primary_darcy_flux) else None
    dq_flux = dlambda_drag if bool(use_primary_darcy_flux) else None
    q_flux_test = lambda_drag_test if bool(use_primary_darcy_flux) else None
    fluid_momentum_scale_c = _as_constant(1.0 if fluid_momentum_scale is None else float(fluid_momentum_scale))
    skeleton_momentum_scale_c = _as_constant(1.0 if skeleton_momentum_scale is None else float(skeleton_momentum_scale))
    pressure_block_lift_scale_c = _as_constant(float(pressure_block_lift_scale))
    total_pressure_ref_c = _as_constant(1.0 if total_pressure_ref is None else float(total_pressure_ref))
    total_pressure_ref_inv_c = _as_constant(1.0 if total_pressure_ref is None else (1.0 / float(total_pressure_ref)))
    drained_bulk_modulus = None
    if total_pressure_ref is not None:
        if _is_seboldt_neo_hookean_model(solid_model_key):
            drained_bulk_modulus = float(lambda_s)
        else:
            drained_bulk_modulus = float(lambda_s) + (2.0 * float(mu_s) / float(dim))
    drained_bulk_over_total_pressure_ref_c = _as_constant(
        1.0 if total_pressure_ref is None else (float(drained_bulk_modulus) / float(total_pressure_ref))
    )
    if support_physics_key == "internal_conversion":
        if float(k_g) != 0.0:
            raise ValueError(
                "support_physics='internal_conversion' conserves alpha, so k_g-driven alpha growth is incompatible."
            )
        if float(D_phi) != 0.0:
            raise ValueError(
                "support_physics='internal_conversion' models the conservative B=alpha(1-phi) balance, "
                "so D_phi regularization is not physically consistent. Keep D_phi=0 and use only the free-fluid gauge."
            )
        if float(phi_cip) != 0.0:
            raise ValueError(
                "support_physics='internal_conversion' currently supports only SUPG stabilization for the conservative "
                "B=alpha(1-phi) balance. The facet-CIP Jacobian is not yet backend-audited to machine precision."
            )
        if float(k_det) != 0.0 or D_det_prev is not None:
            raise ValueError(
                "support_physics='internal_conversion' does not use interface erosion-to-X detachment. "
                "Use damage/cohesion mechanics for sloughing, or switch to support_physics='legacy_exchange'."
            )
        if float(alpha_crack_k) != 0.0 or V_crack_prev is not None:
            raise ValueError(
                "support_physics='internal_conversion' preserves total alpha, so alpha_crack_k/V_crack_prev sinks are incompatible."
            )
        if s_v is not None:
            try:
                if float(s_v) != 0.0:
                    raise ValueError(
                        "support_physics='internal_conversion' requires s_v=0 because growth is already closed through "
                        "the conservative B=alpha(1-phi) balance."
                    )
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(
                    "support_physics='internal_conversion' requires s_v to be omitted or passed as an explicit zero "
                    "scalar/Constant."
                ) from exc
        if ds_v is not None:
            try:
                if float(ds_v) != 0.0:
                    raise ValueError(
                        "support_physics='internal_conversion' requires ds_v=0 because the mass constraint has no "
                        "volumetric source in this mode."
                    )
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(
                    "support_physics='internal_conversion' requires ds_v to be omitted or passed as an explicit zero "
                    "scalar/Constant."
                ) from exc
    if support_physics_key == "stored_support":
        if not bool(use_B_primary):
            raise ValueError("support_physics='stored_support' requires B to be a primary unknown.")
        if float(k_g) != 0.0:
            raise ValueError(
                "support_physics='stored_support' keeps alpha geometric, so k_g-driven alpha growth is incompatible."
            )

    th = _as_constant(theta)
    one_m_th = _c(1.0) - th
    inv_dt = _c(1.0) / dt

    # Optional sources default to 0 (as *expressions*, not test-weighted terms).
    zero_scalar = _c(0.0)
    zero_vector = Constant([0.0] * int(dim), dim=1)
    f_v = f_v if f_v is not None else zero_vector
    f_u = f_u if f_u is not None else zero_vector
    s_v = s_v if s_v is not None else zero_scalar
    # NOTE: `ds_v` is the Gateaux derivative of the volumetric source `s_v`.
    # When no derivative is provided (or a Picard/frozen linearization is used),
    # we must still keep the Jacobian term `-ds_v` in the *trial* role.
    # Using a bare scalar `0` here can leave mixed trial/function metadata in
    # the symbolic pipeline, so use an explicit zero-trial instead.
    if ds_v is None:
        ds_v = zero_scalar * dS
    else:
        # Some drivers pass a frozen/Picard linearization as `Constant(0.0)`.
        # Treat this as the intended "zero derivative" and keep it in the trial
        # role to avoid trial/function shape-mismatch errors during assembly.
        try:
            if float(ds_v) == 0.0:
                ds_v = zero_scalar * dS
        except Exception:
            pass
    if support_physics_key in {"internal_conversion", "stored_support"}:
        s_v = zero_scalar
        ds_v = zero_scalar * dS
    f_phi = f_phi if f_phi is not None else zero_scalar
    f_alpha = f_alpha if f_alpha is not None else zero_scalar
    f_S = f_S if f_S is not None else zero_scalar
    f_X = f_X if f_X is not None else zero_scalar
    g_t_k = g_t_k if g_t_k is not None else zero_vector
    g_t_n = g_t_n if g_t_n is not None else g_t_k
    traction_weight_k = traction_weight_k if traction_weight_k is not None else zero_scalar
    traction_weight_n = traction_weight_n if traction_weight_n is not None else traction_weight_k
    hdiv_tangential_g_k = hdiv_tangential_g_k if hdiv_tangential_g_k is not None else zero_vector
    hdiv_tangential_g_n = hdiv_tangential_g_n if hdiv_tangential_g_n is not None else hdiv_tangential_g_k
    r_skeleton_pressure = None
    a_skeleton_pressure = None
    r_volumetric = None
    a_volumetric = None

    # ------------------------------------------------------------------
    # Skeleton velocity is now a primary unknown (vS).
    # ------------------------------------------------------------------
    div_vS_k = div(vS_k)
    div_vS_n = div(vS_n)
    div_dvS = div(dvS)

    # ------------------------------------------------------------------
    # Optional bulk damage field d: cohesion loss / fracture-like weakening
    # ------------------------------------------------------------------
    use_damage = d_k is not None
    if use_damage:
        if d_n is None or dd is None or d_test is None:
            raise ValueError("d_k provided but one of (d_n, dd, d_test) is missing.")

        kappa_stiff = _c(float(damage_kappa_stiff))
        kappa_perm = _c(float(damage_kappa_perm))
        one_m_kappa_stiff = _one_minus(kappa_stiff)
        one_m_kappa_perm = _one_minus(kappa_perm)

        one_m_d_k = _one_minus(d_k)
        one_m_d_n = _one_minus(d_n)

        # Miehe-type regularized degradation:
        #   g(d) = (1 - κ) (1 - d)^2 + κ,
        # so g(0)=1 and g(1)=κ.
        g_stiff_k = one_m_kappa_stiff * (one_m_d_k * one_m_d_k) + kappa_stiff
        g_stiff_n = one_m_kappa_stiff * (one_m_d_n * one_m_d_n) + kappa_stiff
        dg_stiff_k = (-_c(2.0) * one_m_kappa_stiff * one_m_d_k) * dd

        g_perm_k = one_m_kappa_perm * (one_m_d_k * one_m_d_k) + kappa_perm
        g_perm_n = one_m_kappa_perm * (one_m_d_n * one_m_d_n) + kappa_perm
        dg_perm_k = (-_c(2.0) * one_m_kappa_perm * one_m_d_k) * dd
    else:
        g_stiff_k = _c(1.0)
        g_stiff_n = _c(1.0)
        dg_stiff_k = _c(0.0)
        g_perm_k = _c(1.0)
        g_perm_n = _c(1.0)
        dg_perm_k = _c(0.0)

    # Access fluid-field components through the generic helper so the same
    # code path works for both CG vector spaces and H(div) fields.
    v_k_comp = _components(v_k, dim=int(dim))
    v_n_comp = _components(v_n, dim=int(dim))
    dv_comp = _components(dv, dim=int(dim))
    v_test_comp = _components(v_test, dim=int(dim))

    support_indicator_beta_val = float(support_indicator_beta)
    alpha_support_k = _support_indicator(alpha_k, beta=support_indicator_beta_val)
    alpha_support_n = _support_indicator(alpha_n, beta=support_indicator_beta_val)
    alpha_support_prime_k = _support_indicator_prime(alpha_k, beta=support_indicator_beta_val)
    alpha_support_prime_n = _support_indicator_prime(alpha_n, beta=support_indicator_beta_val)
    alpha_support_second_k = _support_indicator_second(alpha_k, beta=support_indicator_beta_val)
    d_alpha_support_k = alpha_support_prime_k * dalpha
    grad_alpha_support_k = alpha_support_prime_k * grad(alpha_k)
    grad_alpha_support_n = alpha_support_prime_n * grad(alpha_n)
    dgrad_alpha_support_k = (
        alpha_support_second_k * dalpha * grad(alpha_k)
        + alpha_support_prime_k * grad(dalpha)
    )

    # ------------------------------------------------------------------
    # Coefficients (at n/k) and their variations (at k only)
    # ------------------------------------------------------------------
    mu_b_key = str(mu_b_model).strip().lower()
    mu_b_expr = mu_b
    if mu_b_expr is not None and not hasattr(mu_b_expr, "dim"):
        mu_b_expr = _c(float(mu_b_expr))

    if use_B_primary:
        C_k = _C_from_B(B_k)
        C_n = _C_from_B(B_n)
        P_k = _P_from_alpha_B(alpha_k, B_k)
        P_n = _P_from_alpha_B(alpha_n, B_n)
        gradC_k = -grad(B_k)
        gradC_n = -grad(B_n)
        gradB_k = grad(B_k)
        gradB_n = grad(B_n)
        gradP_k = grad(alpha_k) - grad(B_k)
        gradP_n = grad(alpha_n) - grad(B_n)
        dC = -dB
        dC_k = dC
        dB_k = dB
        dP_k = dalpha - dB
        dgradC_k = -grad(dB)
        dgradB_k = grad(dB)
        dgradP_k = grad(dalpha) - grad(dB)
        rho_k = _rho_from_B(B_k, rho_f=rho_f)
        rho_n = _rho_from_B(B_n, rho_f=rho_f)
        mu_k = _mu_from_alpha_B(alpha_k, B_k, mu_f=mu_f, mu_b=mu_b_expr, mu_b_model=mu_b_model)
        mu_n = _mu_from_alpha_B(alpha_n, B_n, mu_f=mu_f, mu_b=mu_b_expr, mu_b_model=mu_b_model)
        drho = rho_f * dC
        if mu_b_key in {"mu", "const", "constant"}:
            dmu = _c(0.0) * dB
        elif mu_b_key in {"phi_mu", "phi*mu", "phi"}:
            dmu = mu_f * dC
        elif mu_b_key in {"alpha_mu", "alpha", "mu_b", "biofilm_mu", "biofilm"}:
            if mu_b_expr is None:
                raise ValueError("mu_b_model='alpha_mu' requires mu_b to be provided.")
            dmu = (mu_b_expr - mu_f) * dalpha
        elif mu_b_key in {"alpha_phi_mu", "alpha_phi", "alpha*phi_mu", "alpha*phi*mu"}:
            if mu_b_expr is None:
                raise ValueError("mu_b_model='alpha_phi_mu' requires mu_b to be provided.")
            dmu = (mu_b_expr - mu_f) * dalpha - mu_b_expr * dB
        else:
            raise ValueError(f"Unknown mu_b_model {mu_b_model!r}.")
        drag_occ_k = P_k
        drag_occ_n = P_n
        ddrag_occ = dP_k
        drag_phi_factor_k = _c(1.0)
        drag_phi_factor_n = _c(1.0)
        ddrag_phi_factor = _c(0.0)
        beta_coeff_k = drag_occ_k * mu_f * g_perm_k
        beta_coeff_n = drag_occ_n * mu_f * g_perm_n
        dbeta_coeff_terms = [mu_f * ddrag_occ * g_perm_k]
        if use_damage:
            dbeta_coeff_damage = drag_occ_k * mu_f * dg_perm_k
            dbeta_coeff_terms.append(dbeta_coeff_damage)
        drag_weight_k = P_k
        drag_weight_n = P_n
        ddrag_weight = dP_k
        porosity_scale_k = P_k
        porosity_scale_n = P_n
        dporosity_scale_k = dP_k
    else:
        rho_k = _rho(alpha_k, phi_k, rho_f=rho_f)
        rho_n = _rho(alpha_n, phi_n, rho_f=rho_f)
        mu_k = _mu(alpha_k, phi_k, mu_f=mu_f, mu_b=mu_b_expr, mu_b_model=mu_b_model)
        mu_n = _mu(alpha_n, phi_n, mu_f=mu_f, mu_b=mu_b_expr, mu_b_model=mu_b_model)
        dC = (phi_k - _c(1.0)) * dalpha + alpha_k * dphi
        drho = rho_f * dC
        if mu_b_key in {"mu", "const", "constant"}:
            dmu = _c(0.0) * dphi
        elif mu_b_key in {"phi_mu", "phi*mu", "phi"}:
            dmu = mu_f * dC
        elif mu_b_key in {"alpha_mu", "alpha", "mu_b", "biofilm_mu", "biofilm"}:
            if mu_b_expr is None:
                raise ValueError("mu_b_model='alpha_mu' requires mu_b to be provided.")
            dmu = (mu_b_expr - mu_f) * dalpha
        elif mu_b_key in {"alpha_phi_mu", "alpha_phi", "alpha*phi_mu", "alpha*phi*mu"}:
            if mu_b_expr is None:
                raise ValueError("mu_b_model='alpha_phi_mu' requires mu_b to be provided.")
            dmu = (phi_k * mu_b_expr - mu_f) * dalpha + (alpha_k * mu_b_expr) * dphi
        else:
            raise ValueError(f"Unknown mu_b_model {mu_b_model!r}.")
        if support_physics_key == "internal_conversion":
            drag_occ_k = alpha_k * _one_minus(phi_k)
            drag_occ_n = alpha_n * _one_minus(phi_n)
            ddrag_occ = _one_minus(phi_k) * dalpha - alpha_k * dphi
        else:
            drag_occ_k = alpha_k
            drag_occ_n = alpha_n
            ddrag_occ = dalpha
        drag_phi_factor_k = phi_k * phi_k
        drag_phi_factor_n = phi_n * phi_n
        ddrag_phi_factor = _c(2.0) * phi_k * dphi
        beta_coeff_k = drag_occ_k * mu_f * drag_phi_factor_k * g_perm_k
        beta_coeff_n = drag_occ_n * mu_f * drag_phi_factor_n * g_perm_n
        dbeta_coeff_terms = [
            mu_f * (ddrag_occ * drag_phi_factor_k) * g_perm_k,
            mu_f * (drag_occ_k * ddrag_phi_factor) * g_perm_k,
        ]
        if use_damage:
            dbeta_coeff_damage = drag_occ_k * mu_f * drag_phi_factor_k * dg_perm_k
            dbeta_coeff_terms.append(dbeta_coeff_damage)
        drag_weight_k = drag_occ_k * drag_phi_factor_k
        drag_weight_n = drag_occ_n * drag_phi_factor_n
        ddrag_weight = ddrag_occ * drag_phi_factor_k + drag_occ_k * ddrag_phi_factor
        C_k = _capacity(alpha_k, phi_k)
        C_n = _capacity(alpha_n, phi_n)
        B_k = alpha_k * _one_minus(phi_k)
        B_n = alpha_n * _one_minus(phi_n)
        P_k = alpha_k * phi_k
        P_n = alpha_n * phi_n
        gradC_k = grad(alpha_k) * (phi_k - _c(1.0)) + grad(phi_k) * alpha_k
        gradC_n = grad(alpha_n) * (phi_n - _c(1.0)) + grad(phi_n) * alpha_n
        gradB_k = grad(alpha_k) * _one_minus(phi_k) - grad(phi_k) * alpha_k
        gradB_n = grad(alpha_n) * _one_minus(phi_n) - grad(phi_n) * alpha_n
        gradP_k = grad(alpha_k) * phi_k + grad(phi_k) * alpha_k
        gradP_n = grad(alpha_n) * phi_n + grad(phi_n) * alpha_n
        dC_k = dC
        dB_k = _one_minus(phi_k) * dalpha - alpha_k * dphi
        dP_k = phi_k * dalpha + alpha_k * dphi
        dgradC_k = grad(dalpha) * (phi_k - _c(1.0)) + grad(alpha_k) * dphi + grad(phi_k) * dalpha + grad(dphi) * alpha_k
        dgradB_k = grad(dalpha) * _one_minus(phi_k) - grad(alpha_k) * dphi - grad(phi_k) * dalpha - grad(dphi) * alpha_k
        dgradP_k = grad(dalpha) * phi_k + grad(alpha_k) * dphi + grad(phi_k) * dalpha + grad(dphi) * alpha_k
        porosity_scale_k = phi_k
        porosity_scale_n = phi_n
        dporosity_scale_k = dphi

    dbeta_coeff = _c(0.0)
    for term in dbeta_coeff_terms:
        dbeta_coeff += term
    diff_k = v_k - vS_k
    diff_n = v_n - vS_n
    ddiff = dv - dvS

    def _dbeta_coeff_weighted(weight_expr):
        acc = _c(0.0)
        for term in dbeta_coeff_terms:
            acc += term * weight_expr
        return acc

    kappa_inv_key = str(kappa_inv_model).strip().lower()

    kc_keys = {
        "kozeny",
        "kozeny_carman",
        "kozeny-carman",
        "kozeny_carman_phi",
        "kc",
    }

    def _kozeny_carman_scale(phi, *, phi_ref, eps_kc):
        # Kozeny–Carman scaling (dimensionless) for inverse permeability:
        #   k^{-1}(phi) ∝ (1-phi)^2 / phi^3
        # We normalize by phi_ref so that k^{-1}(phi_ref) = kappa_inv.
        one_m = _one_minus(phi)
        num = one_m * one_m
        den = phi * phi * phi + eps_kc
        g = num / den

        one_m0 = _one_minus(phi_ref)
        g0 = (one_m0 * one_m0) / (phi_ref * phi_ref * phi_ref + eps_kc)
        return g / g0

    def _d_kozeny_carman_scale(phi, dphi, *, phi_ref, eps_kc):
        # Linearization of the normalized scale: δ[ g(phi)/g(phi_ref) ].
        # g(phi) = (1-phi)^2 / (phi^3+eps)
        one_m = _one_minus(phi)
        num = one_m * one_m
        den = phi * phi * phi + eps_kc

        one_m0 = _one_minus(phi_ref)
        g0 = (one_m0 * one_m0) / (phi_ref * phi_ref * phi_ref + eps_kc)

        # dg/dphi = (num' den - num den') / den^2,  num'=-2(1-phi), den'=3 phi^2
        dg_dphi = ((-_c(2.0) * one_m) * den - num * (_c(3.0) * phi * phi)) / (den * den)
        return (dg_dphi / g0) * dphi

    # Fast/robust path for *scalar spatial* κ^{-1}: keep the legacy scalar formulation.
    if kappa_inv_key in {"spatial", "constant", "const"} and getattr(kappa_inv, "dim", None) == 0:
        drag_mode = "scalar"
        k_inv_k = kappa_inv
        k_inv_n = kappa_inv
        dk_inv_k = _c(0.0)
        beta_k = beta_coeff_k * kappa_inv
        beta_n = beta_coeff_n * kappa_inv
        dbeta = dbeta_coeff * kappa_inv
    elif kappa_inv_key in kc_keys and getattr(kappa_inv, "dim", None) == 0:
        # Kozeny–Carman: permeability depends on porosity (phi).
        drag_mode = "scalar"

        phi_ref_val = float(kappa_inv_phi_ref)
        if not (0.0 < phi_ref_val < 1.0):
            raise ValueError(f"kappa_inv_phi_ref must be in (0,1); got {phi_ref_val}.")
        phi_ref_c = _c(phi_ref_val)
        eps_kc = _c(float(kappa_inv_kc_eps))

        scale_k = _kozeny_carman_scale(porosity_scale_k, phi_ref=phi_ref_c, eps_kc=eps_kc)
        scale_n = _kozeny_carman_scale(porosity_scale_n, phi_ref=phi_ref_c, eps_kc=eps_kc)
        dscale_k = _d_kozeny_carman_scale(porosity_scale_k, dporosity_scale_k, phi_ref=phi_ref_c, eps_kc=eps_kc)

        k_inv_k = kappa_inv * scale_k
        k_inv_n = kappa_inv * scale_n
        dk_inv_k = kappa_inv * dscale_k

        beta_k = beta_coeff_k * k_inv_k
        beta_n = beta_coeff_n * k_inv_n
        dbeta = dbeta_coeff * k_inv_k + beta_coeff_k * dk_inv_k
    else:
        drag_mode = "matrix"

        def _as_invperm_matrix(k_inv, *, dim: int):
            # Accept scalar Constant/Expression as isotropic k_inv * I.
            if getattr(k_inv, "dim", None) == 0:
                return k_inv * Identity(int(dim))
            return k_inv

        K_inv = _as_invperm_matrix(kappa_inv, dim=int(dim))
        if kappa_inv_key in {"refmap", "eulerian_refmap", "eulerian", "reference_map", "reference-map"}:
            k_inv_k = eulerian_k_inv(u_k, K_inv, dim=int(dim))
            k_inv_n = eulerian_k_inv(u_n, K_inv, dim=int(dim))
            dk_inv_k = deulerian_k_inv(u_k, du, K_inv, dim=int(dim))
        elif kappa_inv_key in kc_keys:
            # Kozeny–Carman scaling applied as an isotropic factor to the (possibly anisotropic) K_inv.
            phi_ref_val = float(kappa_inv_phi_ref)
            if not (0.0 < phi_ref_val < 1.0):
                raise ValueError(f"kappa_inv_phi_ref must be in (0,1); got {phi_ref_val}.")
            phi_ref_c = _c(phi_ref_val)
            eps_kc = _c(float(kappa_inv_kc_eps))

            scale_k = _kozeny_carman_scale(porosity_scale_k, phi_ref=phi_ref_c, eps_kc=eps_kc)
            scale_n = _kozeny_carman_scale(porosity_scale_n, phi_ref=phi_ref_c, eps_kc=eps_kc)
            dscale_k = _d_kozeny_carman_scale(porosity_scale_k, dporosity_scale_k, phi_ref=phi_ref_c, eps_kc=eps_kc)

            k_inv_k = scale_k * K_inv
            k_inv_n = scale_n * K_inv
            dk_inv_k = dscale_k * K_inv
        elif kappa_inv_key in {"spatial", "constant", "const"}:
            k_inv_k = K_inv
            k_inv_n = K_inv
            dk_inv_k = None
        else:
            raise ValueError(f"Unknown kappa_inv_model {kappa_inv_model!r}.")

        kdrag_k = _apply_invperm_components(k_inv_k, diff_k, dim=int(dim))
        kdrag_n = _apply_invperm_components(k_inv_n, diff_n, dim=int(dim))

        dkdrag_k = list(_apply_invperm_components(k_inv_k, ddiff, dim=int(dim)))
        if dk_inv_k is not None:
            for i in range(int(dim)):
                for j in range(int(dim)):
                    dkdrag_k[i] += dk_inv_k[i, j] * _vector_component(diff_k, j)
        dkdrag_k = tuple(dkdrag_k)

    split_pore_direct_flux_k = None
    split_pore_direct_flux_d = None
    if bool(use_split_pore_direct_darcy) or bool(use_split_pore_exact_conservative_p) or bool(use_split_pore_exact_total_continuity):
        perm_grad_p_k, dperm_grad_p_k = _apply_perm_from_invperm_components(
            k_inv_k,
            grad(p_pore_k),
            dim=int(dim),
            dinvperm_expr=dk_inv_k,
            dvec_expr=grad(dp_pore),
        )
        split_pore_direct_flux_k = tuple((alpha_support_k / mu_f) * comp for comp in perm_grad_p_k)
        split_pore_direct_flux_d = tuple(
            (d_alpha_support_k / mu_f) * perm_grad_p_k[i] + (alpha_support_k / mu_f) * dperm_grad_p_k[i]
            for i in range(int(dim))
        )

    r_drag_lambda = None
    a_drag_lambda = None

    # Detachment coefficient: lagged by default (depends on previous v/α)
    if D_det_prev is None:
        D_det_prev = _detachment_from_shear_prev(
            v_prev=v_n, alpha_prev=alpha_n, mu_prev=mu_n, k_det=_c(float(k_det)), eta_n=1.0e-12, dim=dim
        )

    # ------------------------------------------------------------------
    # Shared helper quantities for conservative forms (expanded divergence)
    # ------------------------------------------------------------------
    F_free_k = _one_minus(alpha_support_k)
    F_free_n = _one_minus(alpha_support_n)
    gradF_free_k = -grad_alpha_support_k
    gradF_free_n = -grad_alpha_support_n
    dF_free_k = -d_alpha_support_k
    dgradF_free_k = -dgrad_alpha_support_k
    dtimeFfree_k = dF_free_k * inv_dt
    timeFfree_k = (F_free_k - F_free_n) * inv_dt
    divFfree_k = F_free_k * div(v_k) + dot(gradF_free_k, v_k)
    divFfree_n = F_free_n * div(v_n) + dot(gradF_free_n, v_n)
    d_divFfree_k = dF_free_k * div(v_k) + F_free_k * div(dv) + dot(dgradF_free_k, v_k) + dot(gradF_free_k, dv)

    # NS/Darcy liquid momentum coefficients:
    # On the production fluid momentum row we treat the free-fluid phase
    # explicitly, so its inertia/convection carries the free-fluid weight F.
    rho_mom_k = rho_f * F_free_k
    rho_mom_n = rho_f * F_free_n
    drho_mom = rho_f * dF_free_k
    mu_mom_k = mu_f * F_free_k
    mu_mom_n = mu_f * F_free_n
    dmu_mom = mu_f * dF_free_k
    grad_mu_mom_k = mu_f * gradF_free_k
    grad_dmu_mom = mu_f * dgradF_free_k

    gradC_k_components = tuple(gradC_k[i] for i in range(int(dim)))
    gradC_n_components = tuple(gradC_n[i] for i in range(int(dim)))
    timeP_k = (P_k - P_n) * inv_dt
    divBvS_k = B_k * div_vS_k + dot(gradB_k, vS_k)
    divBvS_n = B_n * div_vS_n + dot(gradB_n, vS_n)
    gradB_k_components = tuple(gradB_k[i] for i in range(int(dim)))
    gradB_n_components = tuple(gradB_n[i] for i in range(int(dim)))
    div_alpha_vS_k = alpha_support_k * div_vS_k + dot(grad_alpha_support_k, vS_k)
    div_alpha_vS_n = alpha_support_n * div_vS_n + dot(grad_alpha_support_n, vS_n)
    divCv_k = C_k * div(v_k) + dot(gradC_k, v_k)
    divCv_n = C_n * div(v_n) + dot(gradC_n, v_n)

    divF_k = divCv_k + divBvS_k
    divF_n = divCv_n + divBvS_n
    div_diff_k = div(v_k) - div_vS_k
    div_diff_n = div(v_n) - div_vS_n
    div_ddiff = div(dv) - div_dvS
    divPvS_k = P_k * div_vS_k + dot(gradP_k, vS_k)
    divQ_k = P_k * div_diff_k + dot(gradP_k, diff_k)
    divQ_n = P_n * div_diff_n + dot(gradP_n, diff_n)

    # Jacobian helpers for divCv_k and divBvS_k (k-part only)
    d_div_alpha_vS_k = (
        d_alpha_support_k * div_vS_k
        + alpha_support_k * div_dvS
        + dot(dgrad_alpha_support_k, vS_k)
        + dot(grad_alpha_support_k, dvS)
    )
    d_divCv_k = dC_k * div(v_k) + C_k * div(dv) + dot(dgradC_k, v_k) + dot(gradC_k, dv)
    d_divBvS_k = dB_k * div_vS_k + B_k * div_dvS + dot(dgradB_k, vS_k) + dot(gradB_k, dvS)
    d_timeP_k = dP_k * inv_dt
    d_divPvS_k = dP_k * div_vS_k + P_k * div_dvS + dot(dgradP_k, vS_k) + dot(gradP_k, dvS)
    d_divQ_k = dP_k * div_diff_k + P_k * div_ddiff + dot(dgradP_k, diff_k) + dot(gradP_k, ddiff)

    # Divergence of the test fluxes and their coefficient variations.
    div_alpha_vStest_k = alpha_support_k * div(vS_test) + dot(grad_alpha_support_k, vS_test)
    div_C_vtest_k = C_k * div(v_test) + dot(gradC_k, v_test)
    div_B_vStest_k = B_k * div(vS_test) + dot(gradB_k, vS_test)
    if bool(use_split_pore_pressure) or bool(use_single_pressure_primary_darcy_flux):
        div_Ffree_vtest_k = F_free_k * div(v_test) + dot(gradF_free_k, v_test)
        div_P_vtest_k = P_k * div(v_test) + dot(gradP_k, v_test) if bool(use_split_pore_pressure) else None
    else:
        div_Ffree_vtest_k = None
        div_P_vtest_k = None

    # Expand dgrad·test component-wise to avoid backend-dependent contraction
    # paths for Grad(trial-scalar) · VectorTest.
    d_div_alpha_vStest_k = div(vS_test) * d_alpha_support_k
    d_div_C_vtest_k = div(v_test) * dC_k
    d_div_B_vStest_k = div(vS_test) * dB_k
    if bool(use_split_pore_pressure) or bool(use_single_pressure_primary_darcy_flux):
        d_div_Ffree_vtest_k = div(v_test) * dF_free_k
        d_div_P_vtest_k = div(v_test) * dP_k if bool(use_split_pore_pressure) else None
    else:
        d_div_Ffree_vtest_k = None
        d_div_P_vtest_k = None
    if int(dim) == 2:
        d_div_alpha_vStest_k += (
            dgrad_alpha_support_k[0] * vS_test[0] + dgrad_alpha_support_k[1] * vS_test[1]
        )
        dgradC_k_components = (
            dgradC_k[0],
            dgradC_k[1],
        )
        d_div_C_vtest_k += dgradC_k_components[0] * v_test_comp[0] + dgradC_k_components[1] * v_test_comp[1]

        dgradB_k_components = (
            dgradB_k[0],
            dgradB_k[1],
        )
        d_div_B_vStest_k += dgradB_k_components[0] * vS_test[0] + dgradB_k_components[1] * vS_test[1]
        if bool(use_split_pore_pressure) or bool(use_single_pressure_primary_darcy_flux):
            dgradF_free_k_components = (
                dgradF_free_k[0],
                dgradF_free_k[1],
            )
            d_div_Ffree_vtest_k += (
                dgradF_free_k_components[0] * v_test_comp[0] + dgradF_free_k_components[1] * v_test_comp[1]
            )
        if bool(use_split_pore_pressure):
            dgradP_k_components = (
                dgradP_k[0],
                dgradP_k[1],
            )
            d_div_P_vtest_k += dgradP_k_components[0] * v_test_comp[0] + dgradP_k_components[1] * v_test_comp[1]
    else:
        d_div_alpha_vStest_k += dot(dgrad_alpha_support_k, vS_test)
        d_div_C_vtest_k += dot(dgradC_k, v_test)
        d_div_B_vStest_k += dot(dgradB_k, vS_test)
        if bool(use_split_pore_pressure) or bool(use_single_pressure_primary_darcy_flux):
            d_div_Ffree_vtest_k += dot(dgradF_free_k, v_test)
        if bool(use_split_pore_pressure):
            d_div_P_vtest_k += dot(dgradP_k, v_test)
        dgradC_k_components = tuple(grad(dC_k)[i] for i in range(int(dim)))
        dgradB_k_components = tuple(grad(dB_k)[i] for i in range(int(dim)))
    if bool(use_primary_darcy_flux) and q_flux_test is not None and _is_hdiv_expr(q_flux_test):
        div_alpha_qtest_k = alpha_support_k * div(q_flux_test) + dot(grad_alpha_support_k, q_flux_test)
        d_div_alpha_qtest_k = d_alpha_support_k * div(q_flux_test) + dot(dgrad_alpha_support_k, q_flux_test)
        div_qtest_k = div(q_flux_test)
    else:
        div_alpha_qtest_k = None
        d_div_alpha_qtest_k = None
        div_qtest_k = None
    if use_B_primary:
        if mu_b_key in {"mu", "const", "constant"}:
            grad_mu_k = tuple(zero_scalar for _ in range(int(dim)))
            grad_dmu = tuple(zero_scalar for _ in range(int(dim)))
        elif mu_b_key in {"phi_mu", "phi*mu", "phi"}:
            grad_mu_k = tuple(mu_f * gradC_k[i] for i in range(int(dim)))
            grad_dmu = tuple(mu_f * dgradC_k[i] for i in range(int(dim)))
        elif mu_b_key in {"alpha_mu", "alpha", "mu_b", "biofilm_mu", "biofilm"}:
            coeff_mu = mu_b_expr - mu_f
            grad_mu_k = tuple(coeff_mu * grad(alpha_k)[i] for i in range(int(dim)))
            grad_dmu = tuple(coeff_mu * grad(dalpha)[i] for i in range(int(dim)))
        elif mu_b_key in {"alpha_phi_mu", "alpha_phi", "alpha*phi_mu", "alpha*phi*mu"}:
            coeff_mu = mu_b_expr - mu_f
            grad_mu_k = tuple(coeff_mu * grad(alpha_k)[i] - mu_b_expr * grad(B_k)[i] for i in range(int(dim)))
            grad_dmu = tuple(coeff_mu * grad(dalpha)[i] - mu_b_expr * grad(dB)[i] for i in range(int(dim)))
        else:
            raise ValueError(f"Unknown mu_b_model {mu_b_model!r}.")
    elif mu_b_key in {"mu", "const", "constant"}:
        grad_mu_k = tuple(zero_scalar for _ in range(int(dim)))
        grad_dmu = tuple(zero_scalar for _ in range(int(dim)))
    elif mu_b_key in {"phi_mu", "phi*mu", "phi"}:
        grad_mu_k = tuple(mu_f * ((phi_k - _c(1.0)) * grad(alpha_k)[i] + alpha_k * grad(phi_k)[i]) for i in range(int(dim)))
        grad_dmu = tuple(
            mu_f
            * (
                (phi_k - _c(1.0)) * grad(dalpha)[i]
                + dphi * grad(alpha_k)[i]
                + dalpha * grad(phi_k)[i]
                + alpha_k * grad(dphi)[i]
            )
            for i in range(int(dim))
        )
    elif mu_b_key in {"alpha_mu", "alpha", "mu_b", "biofilm_mu", "biofilm"}:
        coeff_mu = mu_b_expr - mu_f
        grad_mu_k = tuple(coeff_mu * grad(alpha_k)[i] for i in range(int(dim)))
        grad_dmu = tuple(coeff_mu * grad(dalpha)[i] for i in range(int(dim)))
    elif mu_b_key in {"alpha_phi_mu", "alpha_phi", "alpha*phi_mu", "alpha*phi*mu"}:
        coeff_a = phi_k * mu_b_expr - mu_f
        coeff_phi = alpha_k * mu_b_expr
        grad_mu_k = tuple(coeff_a * grad(alpha_k)[i] + coeff_phi * grad(phi_k)[i] for i in range(int(dim)))
        grad_dmu = tuple(
            coeff_a * grad(dalpha)[i]
            + (mu_b_expr * dphi) * grad(alpha_k)[i]
            + coeff_phi * grad(dphi)[i]
            + (mu_b_expr * dalpha) * grad(phi_k)[i]
            for i in range(int(dim))
        )
    else:
        raise ValueError(f"Unknown mu_b_model {mu_b_model!r}.")

    # Shared support coefficient for the split pore-pressure coupling.
    # The whole-domain split keeps a diffuse B-weighted skeleton pressure
    # surrogate, but the pore continuity row follows the constituent-balance
    # reduction and therefore always uses the alpha-weighted support factor.
    if bool(use_split_pore_pressure):
        pore_balance_coeff_c = _as_constant(1.0 if alpha_biot is None else alpha_biot)
        pore_balance_coeff_k = pore_balance_coeff_c * alpha_support_k
        pore_balance_coeff_n = pore_balance_coeff_c * alpha_support_n
        d_pore_balance_coeff_k = pore_balance_coeff_c * d_alpha_support_k
        if skel_press_key == "seboldt":
            pore_biot_coeff_c = _as_constant(1.0 if alpha_biot is None else alpha_biot)
            pore_biot_coeff_k = pore_biot_coeff_c * alpha_support_k
            pore_biot_coeff_n = pore_biot_coeff_c * alpha_support_n
            d_pore_biot_coeff_k = pore_biot_coeff_c * d_alpha_support_k
        elif alpha_biot is not None:
            pore_biot_coeff_c = _as_constant(alpha_biot)
            pore_biot_coeff_k = pore_biot_coeff_c * alpha_support_k
            pore_biot_coeff_n = pore_biot_coeff_c * alpha_support_n
            d_pore_biot_coeff_k = pore_biot_coeff_c * d_alpha_support_k
        else:
            pore_biot_coeff_c = None
            pore_biot_coeff_k = B_k
            pore_biot_coeff_n = B_n
            d_pore_biot_coeff_k = dB_k
    else:
        pore_balance_coeff_c = None
        pore_balance_coeff_k = None
        pore_balance_coeff_n = None
        d_pore_balance_coeff_k = None
        pore_biot_coeff_c = None
        pore_biot_coeff_k = None
        pore_biot_coeff_n = None
        d_pore_biot_coeff_k = None

    # ------------------------------------------------------------------
    # (i) Momentum: conservative one-domain Navier–Stokes / Darcy liquid block
    # ------------------------------------------------------------------
    momdot = (rho_mom_k * v_k - rho_mom_n * v_n) * inv_dt
    fluid_conv_key = str(fluid_convection).strip().lower()
    if fluid_conv_key in {"explicit"}:
        fluid_conv_key = "imex"
    if fluid_conv_key not in {"full", "lagged", "imex", "off"}:
        raise ValueError(
            f"Unknown fluid_convection={fluid_convection!r}. Use 'full' (default), 'lagged', 'imex', or 'off'."
        )
    use_parametric_fluid_convection = any(
        weight is not None
        for weight in (
            fluid_convection_full_weight,
            fluid_convection_lagged_weight,
            fluid_convection_imex_weight,
        )
    )
    if use_parametric_fluid_convection:
        fluid_conv_full_w = _as_scalar_expr(0.0 if fluid_convection_full_weight is None else fluid_convection_full_weight)
        fluid_conv_lagged_w = _as_scalar_expr(
            0.0 if fluid_convection_lagged_weight is None else fluid_convection_lagged_weight
        )
        fluid_conv_imex_w = _as_scalar_expr(0.0 if fluid_convection_imex_weight is None else fluid_convection_imex_weight)
    else:
        fluid_conv_full_w = _c(1.0 if fluid_conv_key == "full" else 0.0)
        fluid_conv_lagged_w = _c(1.0 if fluid_conv_key == "lagged" else 0.0)
        fluid_conv_imex_w = _c(1.0 if fluid_conv_key == "imex" else 0.0)
    hdiv_tangential_method_key = str(hdiv_tangential_method).strip().lower()
    if hdiv_tangential_method_key in {
        "nitsche",
        "consistent",
        "symmetric_nitsche",
        "symmetric-nitsche",
        "fully_consistent",
        "fully-consistent",
    }:
        hdiv_tangential_method_key = "nitsche"
    elif hdiv_tangential_method_key in {"penalty", "penalized"}:
        hdiv_tangential_method_key = "penalty"
    else:
        raise ValueError(f"Unsupported hdiv_tangential_method={hdiv_tangential_method!r}.")

    # Conservative convection: div(ρ v⊗v) = ρ (v·∇)v + v div(ρ v).
    div_rhov_k = rho_f * divFfree_k
    div_rhov_n = rho_f * divFfree_n

    momentum_zero = _c(0.0) * dot(v_k, v_test) * dx
    momentum_terms: dict[str, object] = {
        "transient": inner(momdot, v_test) * dx,
        "convection": momentum_zero,
        "viscous": momentum_zero,
        "pressure": momentum_zero,
        "gamma_div": momentum_zero,
        "drag": momentum_zero,
        "traction": momentum_zero,
        "hdiv_tangential": momentum_zero,
        "body": momentum_zero,
        "supg": momentum_zero,
        "cip": momentum_zero,
    }
    r_mom = momentum_terms["transient"]
    conv_full_k = dot(dot(grad(v_k), v_k), v_test)
    conv_full_n = dot(dot(grad(v_n), v_n), v_test)
    # Picard/IMEX-like linearization:
    #   div(ρ^n v^n ⊗ v^k) = ρ^n (v^n·∇)v^k + v^k div(ρ^n v^n)
    conv_lagged_k = dot(dot(grad(v_k), v_n), v_test)
    conv_lagged_n = dot(dot(grad(v_n), v_n), v_test)
    # IMEX-style explicit convection: treat the convective term fully at the n-level
    # (no v^k dependence). This removes the non-symmetric convection block from the
    # Jacobian and can significantly improve robustness for long transient runs.
    conv_imex_n = conv_full_n
    momentum_terms["convection"] = (
        fluid_conv_full_w * th * (rho_mom_k * conv_full_k + div_rhov_k * dot(v_k, v_test)) * dx
        + fluid_conv_full_w * one_m_th * (rho_mom_n * conv_full_n + div_rhov_n * dot(v_n, v_test)) * dx
        + fluid_conv_lagged_w * th * (rho_mom_n * conv_lagged_k + div_rhov_n * dot(v_k, v_test)) * dx
        + fluid_conv_lagged_w * one_m_th * (rho_mom_n * conv_lagged_n + div_rhov_n * dot(v_n, v_test)) * dx
        + fluid_conv_imex_w * (rho_mom_n * conv_imex_n + div_rhov_n * dot(v_n, v_test)) * dx
    )
    r_mom += momentum_terms["convection"]
    momentum_terms["viscous"] = (
        _c(2.0) * th * mu_mom_k * inner(_epsilon(v_k), _epsilon(v_test)) * dx
        + _c(2.0) * one_m_th * mu_mom_n * inner(_epsilon(v_n), _epsilon(v_test)) * dx
    )
    r_mom += momentum_terms["viscous"]
    # Pressure is treated as a weighted stress, not as a multiplier of the
    # weighted divergence constraint. This keeps the interface band purely
    # stress-driven:
    #   -∫ F p_F div(w)  ->  F grad(p_F) + p_F grad(F)
    # so the free-fluid pressure traction -p_F grad(alpha) is present on the
    # liquid row rather than being lost from the interface transfer.
    if bool(use_split_pore_pressure):
        if bool(use_primary_darcy_flux):
            momentum_terms["pressure"] = -(p_k * div_Ffree_vtest_k) * dx
        else:
            if split_pore_momentum_model_key == "band_alpha":
                if bool(solid_volumetric_split) and skel_press_key == "seboldt":
                    # On the production Seboldt split-pressure branch the
                    # porous pressure traction is carried by the alpha-weighted
                    # support stress (via the volumetric split), not by an extra
                    # liquid-row body-force band.
                    momentum_terms["pressure"] = -(p_k * div_Ffree_vtest_k) * dx
                else:
                    momentum_terms["pressure"] = (
                        -(p_k * div_Ffree_vtest_k)
                        + p_pore_k * dot(grad_alpha_support_k, v_test)
                    ) * dx
            else:
                momentum_terms["pressure"] = -(p_k * div_Ffree_vtest_k + p_pore_k * div_P_vtest_k) * dx
    elif bool(use_single_pressure_primary_darcy_flux):
        momentum_terms["pressure"] = -(p_k * div_Ffree_vtest_k) * dx
    else:
        momentum_terms["pressure"] = -(p_k * div_C_vtest_k) * dx
    r_mom += momentum_terms["pressure"]
    if float(gamma_div) != 0.0 and not bool(use_split_pore_pressure) and not bool(use_single_pressure_primary_darcy_flux):
        gamma_div_c = _as_constant(gamma_div)
        # Consistent augmented-Lagrangian / grad-div stabilization for the mixture
        # volume constraint div(F)=0 with F=C v + B vS.
        momentum_terms["gamma_div"] = gamma_div_c * divF_k * div_C_vtest_k * dx
        r_mom += momentum_terms["gamma_div"]
    if bool(use_primary_darcy_flux):
        momentum_terms["drag"] = momentum_zero
    elif drag_form_key == "mixed_lm":
        momentum_terms["drag"] = _dot_components(lambda_drag_k, v_test, dim=int(dim)) * dx
        r_mom += momentum_terms["drag"]
    elif drag_mode == "scalar":
        momentum_terms["drag"] = (beta_k * dot(v_k, v_test) - beta_k * dot(vS_k, v_test)) * dx
        r_mom += momentum_terms["drag"]
    else:
        momentum_terms["drag"] = beta_coeff_k * _dot_components(kdrag_k, v_test, dim=int(dim)) * dx
        r_mom += momentum_terms["drag"]
    if dGamma is not None:
        momentum_terms["traction"] = -(th * _dot_2d_components(g_t_k, v_test) + one_m_th * _dot_2d_components(g_t_n, v_test)) * dGamma
        r_mom += momentum_terms["traction"]
    if traction_weight_k is not None or traction_weight_n is not None:
        traction_volume = -(
            th * traction_weight_k * _dot_2d_components(g_t_k, v_test)
            + one_m_th * traction_weight_n * _dot_2d_components(g_t_n, v_test)
        ) * dx
        momentum_terms["traction"] = momentum_terms["traction"] + traction_volume
        r_mom += traction_volume
    if ds_hdiv_tangential is not None:
        if int(dim) != 2:
            raise NotImplementedError("Weak tangential H(div) Dirichlet support is currently implemented for 2D only.")
        gamma_t = float(hdiv_tangential_gamma)
        if gamma_t < 0.0:
            raise ValueError("hdiv_tangential_gamma must be nonnegative.")
        n_b = FacetNormal()
        h_b = MeshSize() + _c(1.0e-16)
        v_t_k = _tangential_component_2d(v_k, n_b)
        v_t_n = _tangential_component_2d(v_n, n_b)
        v_t_test = _tangential_component_2d(v_test, n_b)
        g_tan_k = _tangential_component_2d(hdiv_tangential_g_k, n_b)
        g_tan_n = _tangential_component_2d(hdiv_tangential_g_n, n_b)
        penalty_t = _c(gamma_t) / h_b
        gap_t_k = v_t_k - g_tan_k
        gap_t_n = v_t_n - g_tan_n
        momentum_terms["hdiv_tangential"] = (
            penalty_t * (th * mu_mom_k * gap_t_k + one_m_th * mu_mom_n * gap_t_n) * v_t_test
        ) * ds_hdiv_tangential
        if hdiv_tangential_method_key == "nitsche":
            traction_t_k = _tangential_viscous_traction_2d(v_k, mu_mom_k, n_b)
            traction_t_n = _tangential_viscous_traction_2d(v_n, mu_mom_n, n_b)
            traction_t_test_k = _tangential_viscous_traction_2d(v_test, mu_mom_k, n_b)
            traction_t_test_n = _tangential_viscous_traction_2d(v_test, mu_mom_n, n_b)
            momentum_terms["hdiv_tangential"] += (
                -(th * traction_t_k + one_m_th * traction_t_n) * v_t_test
                - (th * traction_t_test_k * gap_t_k + one_m_th * traction_t_test_n * gap_t_n)
            ) * ds_hdiv_tangential
        r_mom += momentum_terms["hdiv_tangential"]
    momentum_terms["body"] = -dot(f_v, v_test) * dx
    r_mom += momentum_terms["body"]

    a_mom = inv_dt * (drho_mom * dot(v_k, v_test) + rho_mom_k * dot(dv, v_test)) * dx

    a_mom += fluid_conv_full_w * th * (
        drho_mom * conv_full_k
        + rho_mom_k * dot(dot(grad(dv), v_k), v_test)
        + rho_mom_k * dot(dot(grad(v_k), dv), v_test)
    ) * dx
    # Jacobian of the conservative correction v div(ρ v) with ρ = ρ_f F.
    a_mom += fluid_conv_full_w * th * ((rho_f * d_divFfree_k) * dot(v_k, v_test)) * dx
    a_mom += fluid_conv_full_w * th * (div_rhov_k * dot(dv, v_test)) * dx
    a_mom += fluid_conv_lagged_w * th * (rho_mom_n * dot(dot(grad(dv), v_n), v_test)) * dx
    a_mom += fluid_conv_lagged_w * th * (div_rhov_n * dot(dv, v_test)) * dx
    a_mom += _c(2.0) * th * (
        dmu_mom * inner(_epsilon(v_k), _epsilon(v_test))
        + mu_mom_k * inner(_epsilon(dv), _epsilon(v_test))
    ) * dx

    # Jacobian of the weighted pressure-stress terms.
    if bool(use_split_pore_pressure):
        if bool(use_primary_darcy_flux):
            a_mom += -(dp * div_Ffree_vtest_k + p_k * d_div_Ffree_vtest_k) * dx
        else:
            a_mom += -(dp * div_Ffree_vtest_k + p_k * d_div_Ffree_vtest_k) * dx
            if split_pore_momentum_model_key == "band_alpha":
                if not (bool(solid_volumetric_split) and skel_press_key == "seboldt"):
                    a_mom += (
                        dp_pore * dot(grad_alpha_support_k, v_test)
                        + p_pore_k * dot(dgrad_alpha_support_k, v_test)
                    ) * dx
            else:
                a_mom += -(dp_pore * div_P_vtest_k + p_pore_k * d_div_P_vtest_k) * dx
    elif bool(use_single_pressure_primary_darcy_flux):
        a_mom += -(dp * div_Ffree_vtest_k + p_k * d_div_Ffree_vtest_k) * dx
    else:
        a_mom += -(dp * div_C_vtest_k + p_k * d_div_C_vtest_k) * dx
    if float(gamma_div) != 0.0 and not bool(use_split_pore_pressure) and not bool(use_single_pressure_primary_darcy_flux):
        gamma_div_c = _as_constant(gamma_div)
        d_divF_k = d_divCv_k + d_divBvS_k
        a_mom += gamma_div_c * (d_divF_k * div_C_vtest_k + divF_k * d_div_C_vtest_k) * dx
    if bool(use_primary_darcy_flux):
        a_mom += _c(0.0) * _dot_components(q_flux_k, v_test, dim=int(dim)) * dx
    elif drag_form_key == "mixed_lm":
        a_mom += _dot_components(dlambda_drag, v_test, dim=int(dim)) * dx
    elif drag_mode == "scalar":
        a_mom += dbeta * (dot(v_k, v_test) - dot(vS_k, v_test)) * dx
        a_mom += beta_k * dot(dv, v_test) * dx
        a_mom += -beta_k * dot(dvS, v_test) * dx
    else:
        a_mom += _dbeta_coeff_weighted(_dot_components(kdrag_k, v_test, dim=int(dim))) * dx
        a_mom += beta_coeff_k * _dot_components(dkdrag_k, v_test, dim=int(dim)) * dx
    if ds_hdiv_tangential is not None:
        n_b = FacetNormal()
        h_b = MeshSize() + _c(1.0e-16)
        v_t_k = _tangential_component_2d(v_k, n_b)
        v_t_test = _tangential_component_2d(v_test, n_b)
        dv_t = _tangential_component_2d(dv, n_b)
        g_tan_k = _tangential_component_2d(hdiv_tangential_g_k, n_b)
        penalty_t = _c(float(hdiv_tangential_gamma)) / h_b
        a_mom += (
            penalty_t * th * (dmu_mom * (v_t_k - g_tan_k) + mu_mom_k * dv_t) * v_t_test
        ) * ds_hdiv_tangential
        if hdiv_tangential_method_key == "nitsche":
            d_traction_t_k = (
                _tangential_viscous_traction_2d(dv, mu_mom_k, n_b)
                + _tangential_viscous_traction_2d(v_k, dmu_mom, n_b)
            )
            traction_t_test_k = _tangential_viscous_traction_2d(v_test, mu_mom_k, n_b)
            d_traction_t_test_k = _tangential_viscous_traction_2d(v_test, dmu_mom, n_b)
            a_mom += (
                -th * d_traction_t_k * v_t_test
                - th * (d_traction_t_test_k * (v_t_k - g_tan_k) + traction_t_test_k * dv_t)
            ) * ds_hdiv_tangential

    # Optional SUPG-like stabilization for the fluid momentum equation.
    #
    # Modes:
    # - "streamline" (legacy): τ ((v^n·∇)v^k, (v^n·∇)w)
    # - "residual":  τ (R_mom,strong, (v^n·∇)w), where R_mom,strong uses the same
    #                 current-iterate momentum operator as the implicit part of the
    #                 time-discrete residual (plus lagged explicit terms for IMEX/lagged
    #                 convection choices). The streamline test direction and τ remain
    #                 lagged for robustness.
    if float(v_supg) != 0.0:
        v_supg_c = _as_constant(v_supg)
        rho_f_val = None
        try:
            rho_f_val = float(rho_f)
        except Exception:
            rho_f_val = None

        # If rho_f==0 then inertia/convection vanishes and this stabilization is irrelevant.
        # Avoid dividing by rho_f in that case.
        if rho_f_val is not None and abs(rho_f_val) < 1.0e-16:
            pass
        else:
            h_v = MeshSize()
            h_v_safe = h_v + _c(1.0e-16)
            h_v2_safe = (h_v * h_v) + _c(1.0e-16)
            adv_w_components = []
            for i in range(int(dim)):
                comp = _c(0.0)
                for j in range(int(dim)):
                    comp += _grad_component(v_test, i, j) * v_n_comp[j]
                adv_w_components.append(comp)
            v_supg_mode_key = str(v_supg_mode or "streamline").strip().lower()
            if v_supg_mode_key in {"streamline", "weak", "legacy"}:
                vmag2 = v_n_comp[0] * v_n_comp[0] + v_n_comp[1] * v_n_comp[1]
                vmag = _sqrt(vmag2 + _c(1.0e-12))
                rho_safe = rho_mom_n + _c(1.0e-16)
                nu_eff = mu_mom_n / rho_safe
                if drag_mode == "scalar":
                    drag_rate = beta_n / rho_safe
                else:
                    # Use the scalar prefactor as the local reaction-rate scale when the
                    # drag operator is anisotropic.
                    drag_rate = beta_coeff_n / rho_safe
                time_scale = _c(2.0) * inv_dt
                adv_scale = _c(2.0) * vmag / h_v_safe
                diff_scale = _c(float(v_supg_c_nu)) * nu_eff / h_v2_safe
                tau_v = v_supg_c / _sqrt(
                    time_scale * time_scale
                    + adv_scale * adv_scale
                    + diff_scale * diff_scale
                    + drag_rate * drag_rate
                    + _c(1.0e-16)
                )
                w_v = _one_minus(alpha_n)  # legacy lagged "fluid-only" localization
                adv_v_k = dot(grad(v_k), v_n)
                adv_w = dot(grad(v_test), v_n)
                momentum_terms["supg"] = tau_v * w_v * inner(adv_v_k, adv_w) * dx
                r_mom += momentum_terms["supg"]
                a_mom += tau_v * w_v * inner(dot(grad(dv), v_n), adv_w) * dx
            elif v_supg_mode_key in {"residual", "strong", "strong_residual", "strong-residual"}:
                # Current-iterate Green's-function tau: use the same k-level
                # transport/reaction coefficients in the residual and Jacobian so the
                # SUPG block is a true Newton linearization, not a quasi-Newton lag.
                vmag2_k = _c(0.0)
                for j in range(int(dim)):
                    vmag2_k += v_k_comp[j] * v_k_comp[j]
                vmag_k = _sqrt(vmag2_k + _c(1.0e-12))
                d_vmag2_k = _c(0.0)
                for j in range(int(dim)):
                    d_vmag2_k += _c(2.0) * v_k_comp[j] * dv_comp[j]
                d_vmag_k = (_c(0.5) / vmag_k) * d_vmag2_k

                rho_safe_k = rho_mom_k + _c(1.0e-16)
                rho_safe_k_sq = rho_safe_k * rho_safe_k
                nu_eff_k = mu_mom_k / rho_safe_k
                d_nu_eff_k = (dmu_mom * rho_safe_k - mu_mom_k * drho_mom) / rho_safe_k_sq
                if drag_mode == "scalar":
                    drag_rate_k = beta_k / rho_safe_k
                    d_drag_rate_k = (dbeta * rho_safe_k - beta_k * drho_mom) / rho_safe_k_sq
                else:
                    drag_rate_k = beta_coeff_k / rho_safe_k
                    d_drag_rate_k = (dbeta_coeff * rho_safe_k - beta_coeff_k * drho_mom) / rho_safe_k_sq

                time_scale = _c(2.0) * inv_dt
                adv_scale_k = _c(2.0) * vmag_k / h_v_safe
                d_adv_scale_k = _c(2.0) * d_vmag_k / h_v_safe
                diff_scale_k = _c(float(v_supg_c_nu)) * nu_eff_k / h_v2_safe
                d_diff_scale_k = _c(float(v_supg_c_nu)) * d_nu_eff_k / h_v2_safe
                tau_denom = (
                    time_scale * time_scale
                    + adv_scale_k * adv_scale_k
                    + diff_scale_k * diff_scale_k
                    + drag_rate_k * drag_rate_k
                    + _c(1.0e-16)
                )
                d_tau_denom = (
                    _c(2.0) * adv_scale_k * d_adv_scale_k
                    + _c(2.0) * diff_scale_k * d_diff_scale_k
                    + _c(2.0) * drag_rate_k * d_drag_rate_k
                )
                tau_v = v_supg_c / _sqrt(tau_denom)
                d_tau_v = -_c(0.5) * tau_v * d_tau_denom / tau_denom
                w_v = _one_minus(alpha_k)
                dw_v = -dalpha

                adv_w_components = []
                d_adv_w_components = []
                for i in range(int(dim)):
                    comp = _c(0.0)
                    dcomp = _c(0.0)
                    for j in range(int(dim)):
                        comp += _grad_component(v_test, i, j) * v_k_comp[j]
                        dcomp += _grad_component(v_test, i, j) * dv_comp[j]
                    adv_w_components.append(comp)
                    d_adv_w_components.append(dcomp)

                strong_visc_k = _strong_div_2mu_eps_components(
                    v_k,
                    mu_mom_k,
                    grad_mu_mom_k,
                    dim=int(dim),
                    hdiv_order=int(fluid_hdiv_order),
                )
                d_strong_visc_k = _d_strong_div_2mu_eps_components(
                    v_k,
                    dv,
                    mu_mom_k,
                    dmu_mom,
                    grad_mu_mom_k,
                    grad_dmu_mom,
                    dim=int(dim),
                    hdiv_order=int(fluid_hdiv_order),
                )
                d_div_rhov_k = rho_f * d_divFfree_k

                strong_mom_comp_k = []
                d_strong_mom_comp_k = []
                for i in range(int(dim)):
                    comp_k = rho_mom_k * (v_k_comp[i] - v_n_comp[i]) * inv_dt
                    dcomp_k = drho_mom * (v_k_comp[i] - v_n_comp[i]) * inv_dt + rho_mom_k * dv_comp[i] * inv_dt

                    if fluid_conv_key == "full":
                        adv_supg_comp = _c(0.0)
                        d_adv_supg_comp = _c(0.0)
                        for j in range(int(dim)):
                            adv_supg_comp += _grad_component(v_k, i, j) * v_k_comp[j]
                            d_adv_supg_comp += _grad_component(dv, i, j) * v_k_comp[j] + _grad_component(v_k, i, j) * dv_comp[j]
                        comp_k += rho_mom_k * adv_supg_comp + div_rhov_k * v_k_comp[i]
                        dcomp_k += drho_mom * adv_supg_comp + rho_mom_k * d_adv_supg_comp + d_div_rhov_k * v_k_comp[i] + div_rhov_k * dv_comp[i]
                    elif fluid_conv_key == "lagged":
                        adv_supg_comp = _c(0.0)
                        d_adv_supg_comp = _c(0.0)
                        for j in range(int(dim)):
                            adv_supg_comp += _grad_component(v_k, i, j) * v_n_comp[j]
                            d_adv_supg_comp += _grad_component(dv, i, j) * v_n_comp[j]
                        comp_k += rho_mom_n * adv_supg_comp + div_rhov_n * v_k_comp[i]
                        dcomp_k += rho_mom_n * d_adv_supg_comp + div_rhov_n * dv_comp[i]
                    elif fluid_conv_key == "imex":
                        adv_supg_comp_n = _c(0.0)
                        for j in range(int(dim)):
                            adv_supg_comp_n += _grad_component(v_n, i, j) * v_n_comp[j]
                        comp_k += rho_mom_n * adv_supg_comp_n + div_rhov_n * v_n_comp[i]

                    comp_k += -strong_visc_k[i]
                    dcomp_k += -d_strong_visc_k[i]

                    # Keep the residual-based stabilization aligned with the
                    # actual momentum residual. Free-fluid pressure now enters
                    # through the weighted stress grad(F p_F). The split
                    # p-only variants then either add the explicit porous
                    # traction band (+p_P grad(alpha)) or the weighted porous
                    # stress grad(P p_P).
                    if bool(use_split_pore_pressure):
                        comp_k += F_free_k * grad(p_k)[i] + p_k * gradF_free_k[i]
                        dcomp_k += dF_free_k * grad(p_k)[i] + F_free_k * grad(dp)[i]
                        dcomp_k += dp * gradF_free_k[i] + p_k * dgradF_free_k[i]
                        if split_pore_momentum_model_key == "band_alpha":
                            if not (bool(solid_volumetric_split) and skel_press_key == "seboldt"):
                                comp_k += p_pore_k * grad_alpha_support_k[i]
                                dcomp_k += dp_pore * grad_alpha_support_k[i]
                                dcomp_k += p_pore_k * dgrad_alpha_support_k[i]
                        else:
                            comp_k += P_k * grad(p_pore_k)[i] + p_pore_k * gradP_k[i]
                            dcomp_k += dP_k * grad(p_pore_k)[i] + P_k * grad(dp_pore)[i]
                            dcomp_k += dp_pore * gradP_k[i] + p_pore_k * dgradP_k[i]
                    else:
                        comp_k += C_k * grad(p_k)[i] + p_k * gradC_k_components[i]
                        dcomp_k += dC * grad(p_k)[i] + C_k * grad(dp)[i]
                        dcomp_k += dp * gradC_k_components[i] + p_k * dgradC_k[i]

                    if drag_form_key == "mixed_lm":
                        # Keep the residual-based SUPG strong form consistent with the
                        # mixed drag formulation used in the main momentum residual.
                        # The stabilization time scale may still use the equivalent
                        # drag-rate coefficient above, but the projected strong residual
                        # itself must carry lambda_drag rather than a direct Brinkman
                        # force beta*(v-vS).
                        comp_k += _vector_component(lambda_drag_k, i)
                        dcomp_k += _vector_component(dlambda_drag, i)
                    elif drag_mode == "scalar":
                        comp_k += beta_k * (v_k_comp[i] - vS_k[i])
                        dcomp_k += dbeta * (v_k_comp[i] - vS_k[i]) + beta_k * (dv_comp[i] - dvS[i])
                    else:
                        comp_k += beta_coeff_k * kdrag_k[i]
                        dcomp_k += _dbeta_coeff_weighted(kdrag_k[i]) + beta_coeff_k * dkdrag_k[i]

                    if traction_weight_k is not None or traction_weight_n is not None:
                        comp_k += -(
                            th * traction_weight_k * _vector_component(g_t_k, i)
                            + one_m_th * traction_weight_n * _vector_component(g_t_n, i)
                        )
                    comp_k += -_vector_component(f_v, i)

                    strong_mom_comp_k.append(comp_k)
                    d_strong_mom_comp_k.append(dcomp_k)

                strong_supg_proj = _c(0.0)
                d_strong_supg_proj = _c(0.0)
                for i in range(int(dim)):
                    strong_supg_proj += strong_mom_comp_k[i] * adv_w_components[i]
                    d_strong_supg_proj += d_strong_mom_comp_k[i] * adv_w_components[i]
                    d_strong_supg_proj += strong_mom_comp_k[i] * d_adv_w_components[i]

                momentum_terms["supg"] = tau_v * w_v * strong_supg_proj * dx
                r_mom += momentum_terms["supg"]
                a_mom += (
                    d_tau_v * w_v * strong_supg_proj
                    + tau_v * dw_v * strong_supg_proj
                    + tau_v * w_v * d_strong_supg_proj
                ) * dx
            else:
                raise ValueError(
                    f"Unknown v_supg_mode={v_supg_mode!r}. "
                    "Use 'streamline' (legacy) or 'residual' (strong-residual SUPG)."
                )

    # Optional CIP (continuous interior penalty) stabilization for the fluid velocity.
    # This is a consistent high-frequency regularization that can improve robustness on coarse meshes.
    if float(v_cip) != 0.0 and ds_cip is not None:
        v_cip_c = _as_constant(v_cip)
        n_int = FacetNormal()
        h_F = avg(MeshSize())
        tau_cip = v_cip_c * (h_F * h_F * h_F) * inv_dt
        w_v = avg(_one_minus(alpha_n))
        momentum_terms["cip"] = tau_cip * w_v * _grad_inner_jump(v_k, v_test, n_int) * ds_cip
        r_mom += momentum_terms["cip"]
        a_mom += tau_cip * w_v * _grad_inner_jump(dv, v_test, n_int) * ds_cip

    # ------------------------------------------------------------------
    # (ii) Mass / volume constraint (expanded divergence)
    # ------------------------------------------------------------------
    # IMPORTANT (time discretization):
    #
    # This is an algebraic (DAE) constraint whose Lagrange multiplier is `p`.
    # Using a θ-average of *only* the divergence while keeping the source fully
    # implicit leads to a mixed time level for the constraint. For θ<1 (e.g.
    # Crank–Nicolson), this can introduce large explicit forcing from the
    # previous step and destabilize the coupled solve (in particular when
    # coupled to stiff substrate kinetics).
    #
    # We therefore enforce the constraint fully implicitly at the k-level:
    #     div(F_k) = α_k s_v(k),
    # consistent with taking the pressure coupling terms at the k-level.
    storativity_c0_c = _c(float(storativity_c0))
    r_total_mass = None
    a_total_mass = None
    if bool(use_split_pore_pressure):
        if fluid_mass_model_key == "fixed_indicator_incompressibility":
            # Prescribed-geometry rigid-interface limit:
            # the free-fluid pressure enforces incompressibility only in the
            # free-fluid fraction F, while the interface transfer is handled by
            # the separate total/pore continuity law rather than by advecting F.
            r_mass = q_test * (F_free_k * div(v_k)) * dx
            a_mass = q_test * (dF_free_k * div(v_k) + F_free_k * div(dv)) * dx
        else:
            # Exact free-fluid content balance for F = 1 - alpha:
            #   ∂_t F + div(F v) = 0.
            #
            # When alpha follows the support kinematics, this is equivalent to the
            # diffuse constituent identity
            #   div((1-alpha) v) - (v - vS) · grad(alpha) = 0.
            r_mass = q_test * (timeFfree_k + divFfree_k) * dx
            a_mass = q_test * (dtimeFfree_k + d_divFfree_k) * dx
        pore_storage_coeff_k = pore_balance_coeff_k
        d_pore_storage_coeff_k = d_pore_balance_coeff_k
        if bool(use_primary_darcy_flux):
            div_q_flux_k = div(q_flux_k)
            div_dq_flux = div(dq_flux)
            if bool(use_split_pore_exact_conservative_q):
                # Exact pore-content balance on the q-primary split branch:
                #   d_t P + div(P vS + q) + beta c0 D_s p_P = alpha s_v.
                #
                # This is the mixed-flux counterpart of the non-q
                # exact_conservative_p branch. It keeps q as the primary Darcy
                # flux while retaining the conservative pore-content transport.
                r_pore = q_pore_test * (timeP_k + divPvS_k + div_q_flux_k - alpha_support_k * s_v) * dx
            elif bool(use_split_pore_exact_total_continuity_q):
                # Exact total fluid-volume continuity on the q-primary split
                # branch:
                #   div(F v + alpha vS + q) + beta c0 D_s p_P = alpha s_v.
                #
                # This is the mixed-flux analogue of the non-q
                # exact_total_continuity branch and exposes the diffuse
                # interface mass-transfer contribution through div(F v).
                r_pore = q_pore_test * (divFfree_k + div_alpha_vS_k + div_q_flux_k - alpha_support_k * s_v) * dx
            else:
                r_pore = q_pore_test * (pore_storage_coeff_k * div_vS_k + div_q_flux_k) * dx
        elif bool(use_split_pore_exact_conservative_p):
            # Exact pore-content balance on the non-q split branch:
            #   d_t P + div(P vS - alpha K/mu grad(p_P)) + beta c0 D_s p_P = alpha s_v,
            # with P = alpha - B.
            r_pore = (
                q_pore_test * (timeP_k + divPvS_k - alpha_support_k * s_v)
                + _dot_components(split_pore_direct_flux_k, grad(q_pore_test), dim=int(dim))
            ) * dx
        elif bool(use_split_pore_exact_total_continuity):
            # Exact total fluid-volume continuity on the non-q split branch:
            #   div(F v + alpha vS - alpha K/mu grad(p_P)) + beta c0 D_s p_P = alpha s_v.
            #
            # Expanding div(F v + alpha vS) exposes the diffuse interface mass
            # transfer term (vS - v) · grad(alpha) directly in the bulk law.
            r_pore = (
                q_pore_test * (divFfree_k + div_alpha_vS_k - alpha_support_k * s_v)
                + _dot_components(split_pore_direct_flux_k, grad(q_pore_test), dim=int(dim))
            ) * dx
        elif bool(use_split_pore_direct_darcy):
            # Direct split-pressure pore closure with q eliminated by Darcy's law:
            #   beta * div(vS) - div(alpha K/mu grad(p_P)) + beta c0 D_s p_P = alpha s_v.
            #
            # This keeps p_P as the primary porous unknown while localizing the
            # Darcy diffusion inside the same diffuse support field alpha that
            # defines the monolithic interface band.
            r_pore = (
                q_pore_test * (pore_storage_coeff_k * div_vS_k - alpha_support_k * s_v)
                + _dot_components(split_pore_direct_flux_k, grad(q_pore_test), dim=int(dim))
            ) * dx
        else:
            # Reduced Biot-type pore-pressure row obtained after eliminating the
            # pore content P through the conservative B-balance:
            #   beta * div(vS) + div(Q_P) + beta * c0 * D_s p_P = 0,
            # with
            #   Q_P = P (v - vS),    P = alpha - B.
            #
            # This continuity coefficient follows the exact constituent-balance
            # reduction and therefore remains alpha-weighted even when the skeleton
            # momentum keeps the diffuse whole-domain pressure-loading surrogate.
            r_pore = q_pore_test * (pore_storage_coeff_k * div_vS_k + divQ_k - alpha_support_k * s_v) * dx
        if float(storativity_c0) != 0.0:
            # After eliminating the pore-content rate, the storage term follows
            # the skeleton material derivative D_s p_P = ∂_t p_P + vS · ∇p_P.
            p_pore_mat_rate_k = ((p_pore_k - p_pore_n) * inv_dt) + dot(vS_k, grad(p_pore_k))
            r_pore += q_pore_test * (pore_storage_coeff_k * storativity_c0_c * p_pore_mat_rate_k) * dx
        if bool(use_primary_darcy_flux):
            if bool(use_split_pore_exact_conservative_q):
                a_pore = q_pore_test * (
                    d_timeP_k
                    + d_divPvS_k
                    + div_dq_flux
                    - d_alpha_support_k * s_v
                    - alpha_support_k * ds_v
                ) * dx
            elif bool(use_split_pore_exact_total_continuity_q):
                a_pore = q_pore_test * (
                    d_divFfree_k
                    + d_div_alpha_vS_k
                    + div_dq_flux
                    - d_alpha_support_k * s_v
                    - alpha_support_k * ds_v
                ) * dx
            else:
                a_pore = q_pore_test * (
                    d_pore_storage_coeff_k * div_vS_k
                    + pore_storage_coeff_k * div_dvS
                    + div_dq_flux
                ) * dx
        elif bool(use_split_pore_exact_conservative_p):
            a_pore = (
                q_pore_test * (
                    d_timeP_k
                    + d_divPvS_k
                    - d_alpha_support_k * s_v
                    - alpha_support_k * ds_v
                )
                + _dot_components(split_pore_direct_flux_d, grad(q_pore_test), dim=int(dim))
            ) * dx
        elif bool(use_split_pore_exact_total_continuity):
            a_pore = (
                q_pore_test * (
                    d_divFfree_k
                    + d_div_alpha_vS_k
                    - d_alpha_support_k * s_v
                    - alpha_support_k * ds_v
                )
                + _dot_components(split_pore_direct_flux_d, grad(q_pore_test), dim=int(dim))
            ) * dx
        elif bool(use_split_pore_direct_darcy):
            a_pore = (
                q_pore_test * (
                    d_pore_storage_coeff_k * div_vS_k
                    + pore_storage_coeff_k * div_dvS
                    - d_alpha_support_k * s_v
                    - alpha_support_k * ds_v
                )
                + _dot_components(split_pore_direct_flux_d, grad(q_pore_test), dim=int(dim))
            ) * dx
        else:
            a_pore = q_pore_test * (
                d_pore_storage_coeff_k * div_vS_k
                + pore_storage_coeff_k * div_dvS
                + d_divQ_k
                - d_alpha_support_k * s_v
                - alpha_support_k * ds_v
            ) * dx
        if float(storativity_c0) != 0.0:
            d_p_pore_mat_rate_k = (dp_pore * inv_dt) + dot(dvS, grad(p_pore_k)) + dot(vS_k, grad(dp_pore))
            a_pore += q_pore_test * (
                storativity_c0_c
                * (
                    d_pore_storage_coeff_k * p_pore_mat_rate_k
                    + pore_storage_coeff_k * d_p_pore_mat_rate_k
                )
            ) * dx
        if bool(use_primary_darcy_flux):
            # Exact total volume identity on the q-primary split branch:
            #   div(F v + alpha vS + q) = alpha s_v.
            # With q = P (v-vS) and C = F + P, this is equivalent to
            #   div(C v + B vS) = alpha s_v.
            r_total_mass = q_pore_test * (divFfree_k + div_alpha_vS_k + div_q_flux_k - alpha_support_k * s_v) * dx
            a_total_mass = q_pore_test * (
                d_divFfree_k
                + d_div_alpha_vS_k
                + div_dq_flux
                - d_alpha_support_k * s_v
                - alpha_support_k * ds_v
            ) * dx
    else:
        if bool(use_single_pressure_primary_darcy_flux):
            div_q_flux_k = div(q_flux_k)
            div_dq_flux = div(dq_flux)
            # q is the Darcy discharge / filtration flux, not the raw slip
            # velocity. With F = 1-alpha, P = alpha-B, C = F+P = 1-B and
            # q = P (v-vS), the exact total continuity law is
            #
            #   div(F v + alpha vS + q) = alpha s_v,
            #
            # which is identically equivalent to
            #
            #   div(C v + B vS) = alpha s_v.
            #
            # One must therefore NOT write div(C v + B vS + q), because that
            # would count the pore-fluid transport twice.
            r_mass = q_test * (divFfree_k + div_alpha_vS_k + div_q_flux_k - alpha_support_k * s_v) * dx
            a_mass = q_test * (
                d_divFfree_k
                + d_div_alpha_vS_k
                + div_dq_flux
                - d_alpha_support_k * s_v
                - alpha_support_k * ds_v
            ) * dx
        else:
            r_mass = q_test * (divF_k - alpha_support_k * s_v) * dx
            if float(storativity_c0) != 0.0:
                # Diffuse-support version of the current-frame Biot storage term
                # alpha * c0 * D_s p, where D_s p = ∂_t p + vS · ∇p.
                p_mat_rate_k = ((p_k - p_n) * inv_dt) + dot(vS_k, grad(p_k))
                r_mass += q_test * (alpha_support_k * storativity_c0_c * p_mat_rate_k) * dx

            # Jacobian of divF_k (k-part only)
            # δC = (φ-1) δα + α δφ
            # δ(α s_v) = (δα) s_v + α (δs_v).
            a_mass = q_test * (d_divCv_k + d_divBvS_k - d_alpha_support_k * s_v - alpha_support_k * ds_v) * dx
            if float(storativity_c0) != 0.0:
                d_p_mat_rate_k = (dp * inv_dt) + dot(dvS, grad(p_k)) + dot(vS_k, grad(dp))
                a_mass += q_test * (
                    storativity_c0_c * (d_alpha_support_k * p_mat_rate_k + alpha_support_k * d_p_mat_rate_k)
                ) * dx
        r_pore = None
        a_pore = None
        r_total_mass = r_mass if bool(use_single_pressure_primary_darcy_flux) else None
        a_total_mass = a_mass if bool(use_single_pressure_primary_darcy_flux) else None

    # ------------------------------------------------------------------
    # (iii) Skeleton momentum (optional inertia + linear/neo-Hookean stress)
    # ------------------------------------------------------------------
    damage_stiff_split_key = str(damage_stiff_split).strip().lower()
    use_miehe_stiff_split = bool(
        use_damage
        and damage_stiff_split_key in {"miehe", "tensile", "tension_compression", "tension-compression", "tc"}
    )

    if use_miehe_stiff_split and solid_model_key not in {
        "linear",
        "small_strain",
        "linear_elastic",
        "hencky",
        "hencky_log",
        "hencky_log_strain",
        "stvk",
        "svk",
        "saint_venant_kirchhoff",
        "saint-venant-kirchhoff",
    }:
        raise ValueError(
            "damage_stiff_split='miehe' is currently only implemented for solid_model in "
            "{'linear','stvk','hencky'} (2D)."
        )
    if use_miehe_stiff_split and int(dim) != 2:
        raise ValueError("damage_stiff_split='miehe' is currently only implemented for dim=2.")

    # Elastic residual/Jacobian contributions.
    #
    # - Default: full-stress degradation via scalar g_stiff(d) multiplier.
    # - Optional: Miehe (tension/compression) split for linear elasticity:
    #     σ = g(d) σ⁺(u) + σ⁻(u),  with σ⁺ built from the positive principal strains.
    #
    # - For finite-strain Hencky hyperelasticity, we implement the classic
    #   Miehe-style tension/compression split at the *energy* level using the
    #   spatial Hencky strain E = log(V). This yields:
    #     ψ⁺ = μ||E⁺||² + (λ/2)⟨tr E⟩₊²,
    #   and the corresponding Cauchy-stress split σ = σ⁺ + σ⁻ is obtained by
    #   differentiating the split energy (Kirchhoff stress conjugate to log strain).
    skeleton_elastic_jac_split_terms: dict[str, object] = {}
    if solid_model_key in {"hencky", "hencky_log", "hencky_log_strain"} and use_miehe_stiff_split:
        eta_pos = float(damage_eta_pos)
        sig_plus_k, sig_minus_k = sigma_hencky_miehe_split(
            u_k, mu_s, lambda_s, dim=int(dim), eta_pos=eta_pos
        )
        sig_plus_n, sig_minus_n = sigma_hencky_miehe_split(
            u_n, mu_s, lambda_s, dim=int(dim), eta_pos=eta_pos
        )
        dsig_plus_k, dsig_minus_k = dsigma_hencky_miehe_split(
            u_k, du, mu_s, lambda_s, dim=int(dim), eta_pos=eta_pos
        )

        r_el_plus_k = inner(sig_plus_k, grad(vS_test))
        r_el_minus_k = inner(sig_minus_k, grad(vS_test))
        r_el_plus_n = inner(sig_plus_n, grad(vS_test))
        r_el_minus_n = inner(sig_minus_n, grad(vS_test))

        a_el_plus = inner(dsig_plus_k, grad(vS_test))
        a_el_minus = inner(dsig_minus_k, grad(vS_test))
    elif solid_model_key in {"stvk", "svk", "saint_venant_kirchhoff", "saint-venant-kirchhoff"} and use_miehe_stiff_split:
        eta_pos = float(damage_eta_pos)
        disc_reg = 1.0e-16
        sig_plus_k, sig_minus_k = sigma_svk_miehe_split(
            u_k, mu_s, lambda_s, dim=int(dim), eta_pos=eta_pos, disc_reg=disc_reg
        )
        sig_plus_n, sig_minus_n = sigma_svk_miehe_split(
            u_n, mu_s, lambda_s, dim=int(dim), eta_pos=eta_pos, disc_reg=disc_reg
        )
        dsig_plus_k, dsig_minus_k = dsigma_svk_miehe_split(
            u_k, du, mu_s, lambda_s, dim=int(dim), eta_pos=eta_pos, disc_reg=disc_reg
        )

        r_el_plus_k = inner(sig_plus_k, grad(vS_test))
        r_el_minus_k = inner(sig_minus_k, grad(vS_test))
        r_el_plus_n = inner(sig_plus_n, grad(vS_test))
        r_el_minus_n = inner(sig_minus_n, grad(vS_test))

        a_el_plus = inner(dsig_plus_k, grad(vS_test))
        a_el_minus = inner(dsig_minus_k, grad(vS_test))
    elif solid_model_key in {"linear", "small_strain", "linear_elastic"} and use_miehe_stiff_split:
        eta_pos = float(damage_eta_pos)
        disc_reg = 1.0e-16
        I = Identity(int(dim))

        # --- k-level split ---
        E_k = _epsilon(u_k)
        E_plus_k, E_minus_k, _, _, _ = spectral_positive_part_2x2_sym(E_k, eta_pos=eta_pos, disc_reg=disc_reg)
        trE_k = div(u_k)
        trE_pos_k = _smooth_pos_u(trE_k, eta=eta_pos)

        sig_plus_k = lambda_s * trE_pos_k * I + _c(2.0) * mu_s * E_plus_k
        sig_minus_k = lambda_s * (trE_k - trE_pos_k) * I + _c(2.0) * mu_s * E_minus_k

        r_el_plus_k = inner(sig_plus_k, grad(vS_test))
        r_el_minus_k = inner(sig_minus_k, grad(vS_test))

        # --- n-level split (lagged, no Jacobian contribution) ---
        E_n = _epsilon(u_n)
        E_plus_n, E_minus_n, _, _, _ = spectral_positive_part_2x2_sym(E_n, eta_pos=eta_pos, disc_reg=disc_reg)
        trE_n = div(u_n)
        trE_pos_n = _smooth_pos_u(trE_n, eta=eta_pos)

        sig_plus_n = lambda_s * trE_pos_n * I + _c(2.0) * mu_s * E_plus_n
        sig_minus_n = lambda_s * (trE_n - trE_pos_n) * I + _c(2.0) * mu_s * E_minus_n

        r_el_plus_n = inner(sig_plus_n, grad(vS_test))
        r_el_minus_n = inner(sig_minus_n, grad(vS_test))

        # --- consistent Jacobian: Gateaux derivatives ---
        dE = _epsilon(du)
        dE_plus = d_spectral_positive_part_2x2_sym(E_k, dE, eta_pos=eta_pos, disc_reg=disc_reg)
        dtrE = div(du)
        dtrE_pos = _smooth_pos_u_prime(trE_k, eta=eta_pos) * dtrE

        dsig_plus_k = lambda_s * dtrE_pos * I + _c(2.0) * mu_s * dE_plus
        dsig_minus_k = lambda_s * (dtrE - dtrE_pos) * I + _c(2.0) * mu_s * (dE - dE_plus)

        a_el_plus = inner(dsig_plus_k, grad(vS_test))
        a_el_minus = inner(dsig_minus_k, grad(vS_test))
    else:
        # Full-stress (legacy) elastic residual/Jacobian.
        mean_dr_k = None
        dmean_dr_k = None
        if solid_model_key in {"linear", "small_strain", "linear_elastic"}:
            if bool(solid_volumetric_split):
                r_el_k = _linear_deviatoric_elastic_term(u_k, vS_test, mu_s=mu_s, dim=int(dim)) + total_pressure_ref_c * pi_s_k * div(vS_test)
                r_el_n = _linear_deviatoric_elastic_term(u_n, vS_test, mu_s=mu_s, dim=int(dim)) + total_pressure_ref_c * pi_s_n * div(vS_test)
                a_el = _linear_deviatoric_elastic_term(du, vS_test, mu_s=mu_s, dim=int(dim)) + total_pressure_ref_c * dpi_s * div(vS_test)
            else:
                r_el_k = _linear_elastic_term(u_k, vS_test, mu_s=mu_s, lambda_s=lambda_s)
                r_el_n = _linear_elastic_term(u_n, vS_test, mu_s=mu_s, lambda_s=lambda_s)
                a_el = _linear_elastic_term(du, vS_test, mu_s=mu_s, lambda_s=lambda_s)
        elif solid_model_key in {"hencky", "hencky_log", "hencky_log_strain"}:
            sig_k = sigma_hencky(u_k, mu_s, lambda_s, dim=int(dim))
            sig_n = sigma_hencky(u_n, mu_s, lambda_s, dim=int(dim))
            r_el_k = inner(sig_k, grad(vS_test))
            r_el_n = inner(sig_n, grad(vS_test))
            dsig_k = dsigma_hencky(u_k, du, mu_s, lambda_s, dim=int(dim))
            a_el = inner(dsig_k, grad(vS_test))
        elif solid_model_key in {"stvk", "svk", "saint_venant_kirchhoff", "saint-venant-kirchhoff"}:
            sig_k = sigma_svk(u_k, mu_s, lambda_s, dim=int(dim))
            sig_n = sigma_svk(u_n, mu_s, lambda_s, dim=int(dim))
            r_el_k = inner(sig_k, grad(vS_test))
            r_el_n = inner(sig_n, grad(vS_test))
            dsig_k = dsigma_svk(u_k, du, mu_s, lambda_s, dim=int(dim))
            a_el = inner(dsig_k, grad(vS_test))
        elif _is_seboldt_neo_hookean_model(solid_model_key):
            sig_k = sigma_neo_hookean_seboldt(u_k, mu_s, lambda_s, dim=int(dim))
            sig_n = sigma_neo_hookean_seboldt(u_n, mu_s, lambda_s, dim=int(dim))
            dsig_k = dsigma_neo_hookean_seboldt(u_k, du, mu_s, lambda_s, dim=int(dim))
            if bool(solid_volumetric_split):
                I = Identity(int(dim))
                mean_dr_k = trace(sig_k) / _c(float(dim))
                mean_dr_n = trace(sig_n) / _c(float(dim))
                dmean_dr_k = trace(dsig_k) / _c(float(dim))
                sig_dev_k = sig_k - mean_dr_k * I
                sig_dev_n = sig_n - mean_dr_n * I
                dsig_dev_k = dsig_k - dmean_dr_k * I
                F_k_dbg = eulerian_F(u_k, dim=int(dim))
                dF_k_dbg = deulerian_F(u_k, du, dim=int(dim))
                J_k_dbg = det(F_k_dbg)
                F_inv_k_dbg = Identity(int(dim)) - grad(u_k)
                dJ_k_dbg = J_k_dbg * trace(dot(F_inv_k_dbg, dF_k_dbg))
                B_k_dbg = dot(F_k_dbg, F_k_dbg.T)
                dB_k_dbg = dot(dF_k_dbg, F_k_dbg.T) + dot(F_k_dbg, dF_k_dbg.T)
                pref_k_dbg = mu_s / J_k_dbg
                dpref_k_dbg = -mu_s * (dJ_k_dbg / (J_k_dbg * J_k_dbg))
                skeleton_elastic_jac_split_terms["pref"] = inner(dpref_k_dbg * (B_k_dbg - I), grad(vS_test))
                skeleton_elastic_jac_split_terms["B"] = inner(pref_k_dbg * dB_k_dbg, grad(vS_test))
                skeleton_elastic_jac_split_terms["vol"] = inner(lambda_s * dJ_k_dbg * I, grad(vS_test))
                skeleton_elastic_jac_split_terms["devsub"] = inner((-dmean_dr_k) * I, grad(vS_test))
                r_el_k = inner(sig_dev_k, grad(vS_test)) + total_pressure_ref_c * pi_s_k * div(vS_test)
                r_el_n = inner(sig_dev_n, grad(vS_test)) + total_pressure_ref_c * pi_s_n * div(vS_test)
                a_el = inner(dsig_dev_k, grad(vS_test)) + total_pressure_ref_c * dpi_s * div(vS_test)
            else:
                r_el_k = inner(sig_k, grad(vS_test))
                r_el_n = inner(sig_n, grad(vS_test))
                a_el = inner(dsig_k, grad(vS_test))
        else:
            # Eulerian reference-map Neo-Hookean stress (Cauchy), compatible with FPI poro Eulerian module.
            if c_nh is None:
                c_nh = mu_s / _c(2.0)
            if beta_nh is None:
                beta_nh = lambda_s / (_c(2.0) * mu_s)

            sig_k = sigma_neo_hookean(u_k, c_nh, beta_nh, dim=int(dim))
            sig_n = sigma_neo_hookean(u_n, c_nh, beta_nh, dim=int(dim))
            r_el_k = inner(sig_k, grad(vS_test))
            r_el_n = inner(sig_n, grad(vS_test))
            dsig_k = dsigma_neo_hookean(u_k, du, c_nh, beta_nh, dim=int(dim))
            a_el = inner(dsig_k, grad(vS_test))

        r_el_plus_k = r_el_k
        r_el_minus_k = _c(0.0)
        r_el_plus_n = r_el_n
        r_el_minus_n = _c(0.0)
        a_el_plus = a_el
        a_el_minus = _c(0.0)

    # Pressure coupling from the support-bearing part of the incompressibility
    # constraint. On the legacy one-pressure branch this is div(B eta). On the
    # one-pressure q-primary branch it becomes the exact alpha-weighted support
    # flux div(alpha eta).
    div_B_vStest_k = B_k * div(vS_test) + dot(gradB_k, vS_test)
    div_B_vStest_n = B_n * div(vS_test) + dot(gradB_n, vS_test)
    press_div_coeff_k = None
    press_div_coeff_n = None
    d_press_div_coeff_k = None
    active_p_k = p_pore_k if bool(use_split_pore_pressure) else p_k
    active_p_n = p_pore_n if bool(use_split_pore_pressure) else p_n
    active_dp = dp_pore if bool(use_split_pore_pressure) else dp
    if bool(use_single_pressure_primary_darcy_flux):
        alpha_biot_c = _as_constant(1.0 if alpha_biot is None else alpha_biot)
        press_coeff_k = alpha_biot_c * alpha_support_k
        press_coeff_n = alpha_biot_c * alpha_support_n
        d_press_coeff_k = alpha_biot_c * d_alpha_support_k
        press_div_coeff_k = press_coeff_k
        press_div_coeff_n = press_coeff_n
        d_press_div_coeff_k = d_press_coeff_k
        active_p_k = p_k
        active_p_n = p_n
        active_dp = dp
        r_skel_press_k = -(alpha_biot_c * active_p_k * div_alpha_vStest_k)
        r_skel_press_n = -(alpha_biot_c * active_p_n * (alpha_support_n * div(vS_test) + dot(grad_alpha_support_n, vS_test)))
        biot_corr_coeff_k = None
    elif skel_press_key == "seboldt":
        if bool(use_split_pore_pressure):
            alpha_biot_c = pore_biot_coeff_c
            press_coeff_k = pore_biot_coeff_k
            press_coeff_n = pore_biot_coeff_n
            d_press_coeff_k = d_pore_biot_coeff_k
        else:
            alpha_biot_c = _as_constant(1.0 if alpha_biot is None else alpha_biot)
            press_coeff_k = alpha_biot_c * alpha_support_k
            press_coeff_n = alpha_biot_c * alpha_support_n
            d_press_coeff_k = alpha_biot_c * d_alpha_support_k
        press_div_coeff_k = press_coeff_k
        press_div_coeff_n = press_coeff_n
        d_press_div_coeff_k = d_press_coeff_k
        r_skel_press_k = -(alpha_biot_c * active_p_k * div_alpha_vStest_k)
        r_skel_press_n = -(alpha_biot_c * active_p_n * (alpha_support_n * div(vS_test) + dot(grad_alpha_support_n, vS_test)))
        biot_corr_coeff_k = None
    else:
        r_skel_press_k = -active_p_k * div_B_vStest_k
        r_skel_press_n = -active_p_n * div_B_vStest_n
        if alpha_biot is not None:
            if bool(use_split_pore_pressure):
                alpha_biot_c = pore_biot_coeff_c
                biot_corr_coeff_k = pore_biot_coeff_k - B_k
                biot_corr_coeff_n = pore_biot_coeff_n - B_n
                d_press_coeff_k = d_pore_biot_coeff_k - dB_k
                press_div_coeff_k = pore_biot_coeff_k
                press_div_coeff_n = pore_biot_coeff_n
                d_press_div_coeff_k = d_pore_biot_coeff_k
            else:
                alpha_biot_c = _as_constant(alpha_biot)
                biot_corr_coeff_k = alpha_biot_c * alpha_support_k - B_k
                biot_corr_coeff_n = alpha_biot_c * alpha_support_n - B_n
                d_press_coeff_k = alpha_biot_c * d_alpha_support_k - dB_k
                press_div_coeff_k = alpha_biot_c * alpha_support_k
                press_div_coeff_n = alpha_biot_c * alpha_support_n
                d_press_div_coeff_k = alpha_biot_c * d_alpha_support_k
            r_skel_press_k += -(active_p_k * biot_corr_coeff_k * div(vS_test))
            r_skel_press_n += -(active_p_n * biot_corr_coeff_n * div(vS_test))
        else:
            alpha_biot_c = None
            biot_corr_coeff_k = None
            d_press_coeff_k = None
            if bool(use_split_pore_pressure):
                press_div_coeff_k = pore_biot_coeff_k
                press_div_coeff_n = pore_biot_coeff_n
                d_press_div_coeff_k = d_pore_biot_coeff_k
            else:
                press_div_coeff_k = B_k
                press_div_coeff_n = B_n
                d_press_div_coeff_k = dB_k
    if bool(solid_volumetric_split) and (
        solid_model_key in {"linear", "small_strain", "linear_elastic"} or _is_seboldt_neo_hookean_model(solid_model_key)
    ):
        vol_pen_c = _as_constant(float(solid_volumetric_penalty))
        split_seboldt_direct_pore_stress = bool(use_split_pore_pressure) and skel_press_key == "seboldt"
        if _is_seboldt_neo_hookean_model(solid_model_key):
            vol_drive_k = pi_s_k - total_pressure_ref_inv_c * mean_dr_k
            d_vol_drive_k = dpi_s - total_pressure_ref_inv_c * dmean_dr_k
            if not split_seboldt_direct_pore_stress:
                vol_drive_k += total_pressure_ref_inv_c * press_div_coeff_k * active_p_k
                d_vol_drive_k += total_pressure_ref_inv_c * (d_press_div_coeff_k * active_p_k + press_div_coeff_k * active_dp)
        else:
            vol_drive_k = pi_s_k - drained_bulk_over_total_pressure_ref_c * div(u_k)
            d_vol_drive_k = dpi_s - drained_bulk_over_total_pressure_ref_c * div(du)
            if not split_seboldt_direct_pore_stress:
                vol_drive_k += total_pressure_ref_inv_c * press_div_coeff_k * active_p_k
                d_vol_drive_k += total_pressure_ref_inv_c * (d_press_div_coeff_k * active_p_k + press_div_coeff_k * active_dp)
        r_volumetric = (
            alpha_support_k * pi_s_test * vol_drive_k
            + vol_pen_c * _one_minus(alpha_support_k) * pi_s_k * pi_s_test
        ) * dx
        a_volumetric = (
            alpha_support_k * pi_s_test * d_vol_drive_k
            + d_alpha_support_k * pi_s_test * vol_drive_k
            + vol_pen_c * ((-d_alpha_support_k) * pi_s_k * pi_s_test + _one_minus(alpha_support_k) * dpi_s * pi_s_test)
        ) * dx
        if float(pressure_block_lift_scale) != 0.0:
            # Add the same constitutive relation to the mass row so the mixed
            # pressure block sees the volumetric variable instead of leaving
            # pi_s isolated in an auxiliary row.
            r_mass += pressure_block_lift_scale_c * (
                alpha_support_k * q_test * vol_drive_k
                + vol_pen_c * _one_minus(alpha_support_k) * pi_s_k * q_test
            ) * dx
            a_mass += pressure_block_lift_scale_c * (
                alpha_support_k * q_test * d_vol_drive_k
                + d_alpha_support_k * q_test * vol_drive_k
                + vol_pen_c * ((-d_alpha_support_k) * pi_s_k * q_test + _one_minus(alpha_support_k) * dpi_s * q_test)
            ) * dx
        if bool(use_single_pressure_primary_darcy_flux):
            r_skel_press_k = -(active_p_k * dot(grad_alpha_support_k, vS_test))
            r_skel_press_n = -(active_p_n * dot(grad_alpha_support_n, vS_test))
        elif skel_press_key == "seboldt":
            # On the split-pressure Seboldt branch, the pore pressure belongs
            # directly in the porous momentum/stress. Keeping it only inside
            # the volumetric constitutive relation leaves the pore traction dead
            # and removes the corresponding grad(alpha) transfer from the
            # alpha-weighted porous stress.
            if split_seboldt_direct_pore_stress:
                r_skel_press_k = -(alpha_biot_c * active_p_k * div_alpha_vStest_k)
                r_skel_press_n = -(alpha_biot_c * active_p_n * (alpha_support_n * div(vS_test) + dot(grad_alpha_support_n, vS_test)))
            else:
                # Keep a test-function factor so the python backend does not see a
                # pure-functional `0 * dx` residual block.
                r_skel_press_k = _c(0.0) * dot(vS_k, vS_test)
                r_skel_press_n = _c(0.0) * dot(vS_n, vS_test)
        else:
            r_skel_press_k = -(active_p_k * dot(gradB_k, vS_test))
            r_skel_press_n = -(active_p_n * dot(gradB_n, vS_test))
    if float(gamma_div) != 0.0 and not bool(use_split_pore_pressure) and not bool(use_single_pressure_primary_darcy_flux):
        gamma_div_c = _as_constant(gamma_div)
        # Consistent augmented-Lagrangian stabilization for the mixture constraint
        # div(F)=0 with F=C v + B vS. The vS variation contributes div(B η).
        r_skel_press_k += gamma_div_c * divF_k * div_B_vStest_k

    r_skel_fluid_interface_traction_k = _c(0.0) * dot(vS_k, vS_test)
    a_skel_fluid_interface_traction = _c(0.0) * dot(dvS, vS_test)
    if bool(use_split_pore_pressure) and (not bool(use_primary_darcy_flux)) and split_pore_momentum_model_key == "band_alpha":
        # The production Seboldt split-pressure branch still needs the explicit
        # free-fluid traction transfer onto the support row; without it the
        # mechanics collapses to the dead rigid state. The porous-pressure band,
        # however, stays inside the alpha-weighted support stress / volumetric
        # split and is not added again on the liquid row above.
        r_skel_fluid_interface_traction_k = p_k * dot(grad_alpha_support_k, vS_test)
        a_skel_fluid_interface_traction = (
            dp * dot(grad_alpha_support_k, vS_test) + p_k * dot(dgrad_alpha_support_k, vS_test)
        )

    # drag reaction: -β (v - vS)
    # Since beta already contains α, if we use alpha again then it would square and it won't 
    # be equal to the drag from the momentum of the fluid.
    if bool(use_primary_darcy_flux):
        r_skel_drag_k = _c(0.0) * dot(vS_k, vS_test)
        r_skel_drag_n = _c(0.0) * dot(vS_n, vS_test)
    elif drag_form_key == "mixed_lm":
        r_skel_drag_k = -_dot_components(lambda_drag_k, vS_test, dim=int(dim))
        r_skel_drag_n = -_dot_components(lambda_drag_n, vS_test, dim=int(dim))
    elif drag_mode == "scalar":
        r_skel_drag_k = -beta_k * dot(v_k - vS_k, vS_test)
        r_skel_drag_n = -beta_n * dot(v_n - vS_n, vS_test)
    else:
        r_skel_drag_k = -beta_coeff_k * _dot_components(kdrag_k, vS_test, dim=int(dim))
        r_skel_drag_n = -beta_coeff_n * _dot_components(kdrag_n, vS_test, dim=int(dim))

    # Time discretization for the (quasi-static) skeleton momentum balance.
    #
    # When `include_skeleton_acceleration=False`, the skeleton equation is an
    # algebraic equilibrium (no time derivative). A θ-average would introduce
    # explicit forcing from the previous step unless *all* terms (including the
    # pressure coupling) are treated consistently. For robustness, we therefore
    # evaluate the quasi-static balance fully at the k-level.
    sk_th = th if bool(include_skeleton_acceleration) else _c(1.0)
    sk_one_m_th = one_m_th if bool(include_skeleton_acceleration) else _c(0.0)

    skeleton_terms: dict[str, object] = {}
    skeleton_terms["elastic_k"] = sk_th * alpha_support_k * (g_stiff_k * r_el_plus_k + r_el_minus_k) * dx
    skeleton_terms["elastic_n"] = sk_one_m_th * alpha_support_n * (g_stiff_n * r_el_plus_n + r_el_minus_n) * dx
    skeleton_terms["pressure"] = (sk_th * r_skel_press_k + sk_one_m_th * r_skel_press_n) * dx
    skeleton_terms["drag"] = (sk_th * r_skel_drag_k + sk_one_m_th * r_skel_drag_n) * dx
    skeleton_terms["fluid_interface_traction"] = sk_th * r_skel_fluid_interface_traction_k * dx
    r_skeleton = (
        skeleton_terms["elastic_k"]
        + skeleton_terms["elastic_n"]
        + skeleton_terms["pressure"]
        + skeleton_terms["drag"]
        + skeleton_terms["fluid_interface_traction"]
    )
    r_skeleton_pressure = skeleton_terms["pressure"]
    # External body force is weighted by biofilm presence α, but not degraded by g_stiff(d).
    skeleton_terms["body"] = -dot(alpha_support_k * f_u, vS_test) * dx
    r_skeleton += skeleton_terms["body"]
    if dGamma is not None:
        skeleton_terms["traction_boundary"] = (
            th * _dot_2d_components(g_t_k, vS_test) + one_m_th * _dot_2d_components(g_t_n, vS_test)
        ) * dGamma
        r_skeleton += skeleton_terms["traction_boundary"]
    if traction_weight_k is not None or traction_weight_n is not None:
        skeleton_terms["traction_volume"] = (
            th * traction_weight_k * _dot_2d_components(g_t_k, vS_test)
            + one_m_th * traction_weight_n * _dot_2d_components(g_t_n, vS_test)
        ) * dx
        r_skeleton += skeleton_terms["traction_volume"]

    # Extension / stabilization coefficients.
    # - `gamma_u` controls u-extension in the kinematic constraint below.
    # - vS also needs an extension penalty in the free-fluid region (α≈0) to keep
    #   the one-domain CG formulation well-posed (otherwise vS DOFs in pure fluid
    #   elements can become weakly constrained / singular).
    gamma_u_c = _as_constant(gamma_u)
    if regularization_weight is None:
        reg_weight_k = _c(1.0)
    else:
        try:
            reg_weight_k = _c(float(regularization_weight))
        except Exception:
            reg_weight_k = regularization_weight
    reg_weight_cip = avg(reg_weight_k) if ds_cip is not None else None
    gamma_vS_input = gamma_u if gamma_vS is None else gamma_vS
    gamma_vS_eff = float(gamma_vS_input)
    vS_ext_mode = str(u_extension_mode if vS_extension_mode is None else vS_extension_mode).strip().lower()
    gamma_vS_c = _as_constant(gamma_vS_input)
    # Mirror u-extension pinning by default: both u-extension (grad-mode) and the
    # vS-extension (grad-mode) have a global-translation nullspace in a one-domain
    # CG setting. A tiny L2 pin in the fluid region breaks the nullspace and
    # materially improves Newton robustness without affecting the biofilm region
    # (the pin weight scales like (1-α)^2).
    if gamma_vS_pin is None and vS_ext_mode in {"grad", "h1"} and float(gamma_u_pin) != 0.0:
        gamma_vS_pin_input = gamma_u_pin
    else:
        gamma_vS_pin_input = 0.0 if gamma_vS_pin is None else gamma_vS_pin
    gamma_vS_pin_eff = float(gamma_vS_pin_input)
    gamma_vS_pin_c = _as_constant(gamma_vS_pin_input)
    vS_pin_pow = int(gamma_vS_pin_power)
    if vS_pin_pow < 1:
        raise ValueError(f"gamma_vS_pin_power must be >= 1; got {gamma_vS_pin_power}.")
    a_skel_visco_alpha = None
    a_skel_visco_vS = None
    skeleton_jac_terms: dict[str, object] = {}
    if float(solid_visco_eta) != 0.0:
        eta_s_c = _c(float(solid_visco_eta))
        # grad_vS_k = (grad(u_k) - grad(u_n))/dt
        # eps_vS_k = 0.5*(grad_vS_k + grad_vS_k.T)
        sig_visc_k = _c(2.0) * eta_s_c * _epsilon(vS_k)
        sig_visc_n = _c(2.0) * eta_s_c * _epsilon(vS_n)
        r_visc_k = inner(sig_visc_k, grad(vS_test))
        r_visc_n = inner(sig_visc_n, grad(vS_test))
        # Treat viscosity as part of the skeleton response: localize by α and apply the same stiffness
        # degradation g_stiff(d) used for elasticity.
        skeleton_terms["viscosity"] = (th * alpha_support_k * g_stiff_k * r_visc_k + one_m_th * alpha_support_n * g_stiff_n * r_visc_n) * dx
        r_skeleton += skeleton_terms["viscosity"]

        # Consistent k-part Jacobian: δ[α g σ_visc(vS)] = δ(α g) r_visc + α g a_visc.
        #
        # IMPORTANT: keep trial-family contributions separated. Mixing dalpha (and/or dd)
        # with dvS in a single integrand sum can trigger VecOpInfo shape-mismatch errors
        # in the current compiler/assembler pipeline.
        sig_dvisc = _c(2.0) * eta_s_c * _epsilon(dvS)
        a_visc = inner(sig_dvisc, grad(vS_test))
        if use_damage:
            w_ag = d_alpha_support_k * g_stiff_k + alpha_support_k * dg_stiff_k
        else:
            # Avoid mixing trial and function roles in the symbolic pipeline:
            # when damage is disabled, dg_stiff_k is exactly 0 but still carries
            # "function" metadata, and (trial + 0*function) can break assembly.
            w_ag = d_alpha_support_k * g_stiff_k
        a_skel_visco_alpha = sk_th * (r_visc_k * w_ag) * dx
        a_skel_visco_vS = sk_th * (alpha_support_k * g_stiff_k * a_visc) * dx

    # Optional extension penalty for vS in the free-fluid region.
    if float(gamma_vS_eff) != 0.0:
        if vS_ext_mode in {"l2", "mass"}:
            h_u = MeshSize()
            inv_h2 = _c(1.0) / (h_u * h_u)
            skeleton_terms["extension"] = reg_weight_k * gamma_vS_c * inv_h2 * _one_minus(alpha_support_k) * dot(vS_k, vS_test) * dx
            r_skeleton += skeleton_terms["extension"]
        elif vS_ext_mode in {"grad", "h1"}:
            # Gradient penalty does not fight rigid translations (∇vS≈0).
            skeleton_terms["extension"] = reg_weight_k * gamma_vS_c * _one_minus(alpha_support_k) * inner(grad(vS_k), grad(vS_test)) * dx
            r_skeleton += skeleton_terms["extension"]
            if float(gamma_vS_pin_eff) != 0.0:
                h_u = MeshSize()
                inv_h2 = _c(1.0) / (h_u * h_u)
                w_pin = _one_minus(alpha_support_k)
                w_pin_pow = w_pin
                for _ in range(vS_pin_pow - 1):
                    w_pin_pow = w_pin_pow * w_pin
                skeleton_terms["extension_pin"] = reg_weight_k * gamma_vS_pin_c * inv_h2 * w_pin_pow * dot(vS_k, vS_test) * dx
                r_skeleton += skeleton_terms["extension_pin"]
        else:
            raise ValueError(f"Unknown vS_extension_mode {vS_extension_mode!r}.")

    # Jacobian contributions (k-part only)
    if bool(use_primary_darcy_flux):
        drag_term_k = _c(0.0) * dot(vS_k, vS_test)
        d_drag_term_k = _c(0.0) * dot(dvS, vS_test)
    elif drag_form_key == "mixed_lm":
        drag_term_k = -_dot_components(lambda_drag_k, vS_test, dim=int(dim))
        d_drag_term_k = -_dot_components(dlambda_drag, vS_test, dim=int(dim))
    elif drag_mode == "scalar":
        drag_term_k = -beta_k * dot(v_k, vS_test) + beta_k * dot(vS_k, vS_test)
        d_drag_term_k = -dbeta * (dot(v_k, vS_test) - dot(vS_k, vS_test))
        d_drag_term_k += -beta_k * (dot(dv, vS_test) - dot(dvS, vS_test))
    else:
        drag_term_k = -beta_coeff_k * _dot_components(kdrag_k, vS_test, dim=int(dim))
        d_drag_term_k = -(
            _dbeta_coeff_weighted(_dot_components(kdrag_k, vS_test, dim=int(dim)))
            + beta_coeff_k * _dot_components(dkdrag_k, vS_test, dim=int(dim))
        )

    # Jacobian of the elastic part (k-part only).
    #
    # For the default full-stress model, this reduces to:
    #   δ[α g(d) r_el(u)] = α g a_el + δ(α g) r_el.
    #
    # For Miehe split (linear elasticity), g(d) multiplies only the tensile part:
    #   δ[α (g r⁺ + r⁻)] = α (δg r⁺ + g δr⁺ + δr⁻) + δα (g r⁺ + r⁻).
    elastic_jac_k = g_stiff_k * a_el_plus + a_el_minus
    if use_damage:
        # Only include δg(d)·r⁺ when the damage field is part of the unknown vector.
        # Otherwise this term would appear as a test-only contribution in the Jacobian
        # (0·test) and break matrix assembly in the python backend.
        elastic_jac_k += dg_stiff_k * r_el_plus_k

    elastic_alpha_drive_k = (g_stiff_k * r_el_plus_k + r_el_minus_k)
    skeleton_jac_terms["elastic"] = sk_th * (alpha_support_k * elastic_jac_k + elastic_alpha_drive_k * d_alpha_support_k) * dx
    if skeleton_elastic_jac_split_terms:
        for key, term in tuple(skeleton_elastic_jac_split_terms.items()):
            skeleton_jac_terms[f"elastic_{key}"] = sk_th * alpha_support_k * term * dx
    a_skel = skeleton_jac_terms["elastic"]
    if a_skel_visco_alpha is not None:
        skeleton_jac_terms["viscosity_alpha"] = a_skel_visco_alpha
        a_skel += a_skel_visco_alpha
    if a_skel_visco_vS is not None:
        skeleton_jac_terms["viscosity_vS"] = a_skel_visco_vS
        a_skel += a_skel_visco_vS
    # Jacobian of the pressure coupling.
    if bool(solid_volumetric_split) and (
        solid_model_key in {"linear", "small_strain", "linear_elastic"} or _is_seboldt_neo_hookean_model(solid_model_key)
    ):
        if bool(use_single_pressure_primary_darcy_flux):
            a_skel_pressure = sk_th * (
                -(active_dp * dot(grad_alpha_support_k, vS_test) + active_p_k * dot(dgrad_alpha_support_k, vS_test))
            ) * dx
        elif skel_press_key == "seboldt":
            if bool(use_split_pore_pressure):
                a_skel_pressure = sk_th * (
                    -(alpha_biot_c * (active_dp * div_alpha_vStest_k + active_p_k * d_div_alpha_vStest_k))
                ) * dx
            else:
                # Preserve both trial and test roles for the identically-zero
                # split-Seboldt remainder so the python backend can assemble it.
                a_skel_pressure = _c(0.0) * dot(dvS, vS_test) * dx
        else:
            a_skel_pressure = sk_th * (-(active_dp * dot(gradB_k, vS_test) + active_p_k * dot(dgradB_k, vS_test))) * dx
    elif bool(use_single_pressure_primary_darcy_flux):
        a_skel_pressure = sk_th * (
            -(alpha_biot_c * (active_dp * div_alpha_vStest_k + active_p_k * d_div_alpha_vStest_k))
        ) * dx
    elif skel_press_key == "seboldt":
        a_skel_pressure = sk_th * (
            -(alpha_biot_c * (active_dp * div_alpha_vStest_k + active_p_k * d_div_alpha_vStest_k))
        ) * dx
    else:
        a_skel_pressure = sk_th * (-(active_dp * div_B_vStest_k + active_p_k * d_div_B_vStest_k)) * dx
        if alpha_biot_c is not None:
            a_skel_pressure += sk_th * (-(active_dp * biot_corr_coeff_k + active_p_k * d_press_coeff_k) * div(vS_test)) * dx
    if float(gamma_div) != 0.0 and not bool(use_split_pore_pressure) and not bool(use_single_pressure_primary_darcy_flux):
        gamma_div_c = _as_constant(gamma_div)
        d_divF_k = d_divCv_k + d_divBvS_k
        a_skel_pressure += sk_th * gamma_div_c * (d_divF_k * div_B_vStest_k + divF_k * d_div_B_vStest_k) * dx
    skeleton_jac_terms["pressure"] = a_skel_pressure
    a_skel += a_skel_pressure
    skeleton_jac_terms["fluid_interface_traction"] = sk_th * a_skel_fluid_interface_traction * dx
    a_skel += skeleton_jac_terms["fluid_interface_traction"]
    # Drag term is *not* multiplied by alpha again: beta already contains alpha (one-domain blend).
    skeleton_jac_terms["drag"] = sk_th * d_drag_term_k * dx
    a_skel += skeleton_jac_terms["drag"]
    skeleton_jac_terms["body"] = -(dot(f_u, vS_test) * d_alpha_support_k) * dx
    a_skel += skeleton_jac_terms["body"]
    if bool(use_primary_darcy_flux):
        q_darcy_k = tuple(mu_f * comp for comp in _apply_invperm_components(k_inv_k, q_flux_k, dim=int(dim)))
        q_darcy_dq = tuple(mu_f * comp for comp in _apply_invperm_components(k_inv_k, dq_flux, dim=int(dim)))
        if dk_inv_k is None:
            q_darcy_dk = tuple(_c(0.0) * _vector_component(q_flux_k, i) for i in range(int(dim)))
        else:
            q_darcy_dk = tuple(mu_f * comp for comp in _apply_invperm_components(dk_inv_k, q_flux_k, dim=int(dim)))
        q_extension_k = tuple(_one_minus(alpha_support_k) * _vector_component(q_flux_k, i) for i in range(int(dim)))
        q_extension_d = tuple(
            (-d_alpha_support_k) * _vector_component(q_flux_k, i) + _one_minus(alpha_support_k) * _vector_component(dq_flux, i)
            for i in range(int(dim))
        )
        if bool(use_split_pore_pressure) and div_alpha_qtest_k is not None and d_div_alpha_qtest_k is not None:
            # True mixed Darcy weak form for an H(div) primary flux q:
            #   \int alpha * mu K^{-1} q · z
            # - \int p_P div(alpha z)
            # + boundary/interface terms handled separately.
            #
            # The top Dirichlet pore-pressure condition contributes zero on this
            # benchmark branch (p_P=0 there). The diffuse interface closures may
            # then add the remaining boundary power consistently on z·n.
            darcy_flux_bulk_k = tuple(alpha_support_k * q_darcy_k[i] for i in range(int(dim)))
            darcy_flux_bulk_d = tuple(
                d_alpha_support_k * q_darcy_k[i] + alpha_support_k * (q_darcy_dk[i] + q_darcy_dq[i])
                for i in range(int(dim))
            )
            r_drag_lambda = (
                _dot_components(darcy_flux_bulk_k, q_flux_test, dim=int(dim))
                - p_pore_k * div_alpha_qtest_k
                + _dot_components(q_extension_k, q_flux_test, dim=int(dim))
            ) * dx
            a_drag_lambda = (
                _dot_components(darcy_flux_bulk_d, q_flux_test, dim=int(dim))
                - (dp_pore * div_alpha_qtest_k + p_pore_k * d_div_alpha_qtest_k)
                + _dot_components(q_extension_d, q_flux_test, dim=int(dim))
            ) * dx
        elif bool(use_split_pore_pressure):
            darcy_bulk_k = tuple(alpha_support_k * (q_darcy_k[i] + grad(p_pore_k)[i]) for i in range(int(dim)))
            darcy_bulk_d = tuple(
                d_alpha_support_k * (q_darcy_k[i] + grad(p_pore_k)[i])
                + alpha_support_k * (q_darcy_dk[i] + q_darcy_dq[i] + grad(dp_pore)[i])
                for i in range(int(dim))
            )
            r_drag_lambda = (
                _dot_components(darcy_bulk_k, q_flux_test, dim=int(dim))
                + _dot_components(q_extension_k, q_flux_test, dim=int(dim))
            ) * dx
            a_drag_lambda = (
                _dot_components(darcy_bulk_d, q_flux_test, dim=int(dim))
                + _dot_components(q_extension_d, q_flux_test, dim=int(dim))
            ) * dx
        elif div_qtest_k is not None:
            darcy_flux_bulk_k = tuple(alpha_support_k * q_darcy_k[i] for i in range(int(dim)))
            darcy_flux_bulk_d = tuple(
                d_alpha_support_k * q_darcy_k[i] + alpha_support_k * (q_darcy_dk[i] + q_darcy_dq[i])
                for i in range(int(dim))
            )
            r_drag_lambda = (
                _dot_components(darcy_flux_bulk_k, q_flux_test, dim=int(dim))
                - p_k * div_qtest_k
                + _dot_components(q_extension_k, q_flux_test, dim=int(dim))
            ) * dx
            a_drag_lambda = (
                _dot_components(darcy_flux_bulk_d, q_flux_test, dim=int(dim))
                - dp * div_qtest_k
                + _dot_components(q_extension_d, q_flux_test, dim=int(dim))
            ) * dx
        else:
            darcy_bulk_k = tuple(alpha_support_k * (q_darcy_k[i] + grad(p_k)[i]) for i in range(int(dim)))
            darcy_bulk_d = tuple(
                d_alpha_support_k * (q_darcy_k[i] + grad(p_k)[i])
                + alpha_support_k * (q_darcy_dk[i] + q_darcy_dq[i] + grad(dp)[i])
                for i in range(int(dim))
            )
            r_drag_lambda = (
                _dot_components(darcy_bulk_k, q_flux_test, dim=int(dim))
                + _dot_components(q_extension_k, q_flux_test, dim=int(dim))
            ) * dx
            a_drag_lambda = (
                _dot_components(darcy_bulk_d, q_flux_test, dim=int(dim))
                + _dot_components(q_extension_d, q_flux_test, dim=int(dim))
            ) * dx
    elif drag_form_key == "mixed_lm":
        if drag_mode == "scalar":
            if kappa_inv_key in kc_keys and getattr(kappa_inv, "dim", None) == 0:
                drag_core_k = mu_f * k_inv_k
                ddrag_core_inv_k = -(mu_f * dk_inv_k) / (drag_core_k * drag_core_k)
            else:
                drag_core_k = mu_f * kappa_inv
                ddrag_core_inv_k = _c(0.0)
            drag_core_inv_k = _c(1.0) / drag_core_k
            r_drag_lambda = dot((drag_core_inv_k * lambda_drag_k) - (drag_weight_k * diff_k), lambda_drag_test) * dx
            a_drag_lambda = dot(
                (ddrag_core_inv_k * lambda_drag_k)
                + (drag_core_inv_k * dlambda_drag)
                - (ddrag_weight * diff_k)
                - (drag_weight_k * ddiff),
                lambda_drag_test,
            ) * dx
        else:
            if int(dim) != 2:
                raise NotImplementedError("drag_formulation='mixed_lm' is currently implemented for matrix drag only in 2D.")
            det_k = k_inv_k[0, 0] * k_inv_k[1, 1] - k_inv_k[0, 1] * k_inv_k[1, 0]
            lam_core_inv_raw_k = (
                (k_inv_k[1, 1] * _vector_component(lambda_drag_k, 0) - k_inv_k[0, 1] * _vector_component(lambda_drag_k, 1)) / det_k,
                (-k_inv_k[1, 0] * _vector_component(lambda_drag_k, 0) + k_inv_k[0, 0] * _vector_component(lambda_drag_k, 1)) / det_k,
            )
            lam_core_inv_k = (
                (_c(1.0) / mu_f) * lam_core_inv_raw_k[0],
                (_c(1.0) / mu_f) * lam_core_inv_raw_k[1],
            )
            lam_core_inv_dlam_k = (
                (_c(1.0) / mu_f)
                * ((k_inv_k[1, 1] * _vector_component(dlambda_drag, 0) - k_inv_k[0, 1] * _vector_component(dlambda_drag, 1)) / det_k),
                (_c(1.0) / mu_f)
                * ((-k_inv_k[1, 0] * _vector_component(dlambda_drag, 0) + k_inv_k[0, 0] * _vector_component(dlambda_drag, 1)) / det_k),
            )
            if dk_inv_k is None:
                d_lam_core_inv_k = tuple(_c(0.0) * _vector_component(lambda_drag_k, i) for i in range(int(dim)))
            else:
                dcore_rhs_k = _apply_invperm_components(dk_inv_k, lam_core_inv_raw_k, dim=int(dim))
                d_lam_core_inv_k = (
                    (_c(1.0) / mu_f) * (-(k_inv_k[1, 1] * dcore_rhs_k[0] - k_inv_k[0, 1] * dcore_rhs_k[1]) / det_k),
                    (_c(1.0) / mu_f) * (-(-k_inv_k[1, 0] * dcore_rhs_k[0] + k_inv_k[0, 0] * dcore_rhs_k[1]) / det_k),
                )
            d_lam_core_inv_k = tuple(d_lam_core_inv_k[i] + lam_core_inv_dlam_k[i] for i in range(int(dim)))
            r_drag_lambda = (
                _dot_components(lam_core_inv_k, lambda_drag_test, dim=int(dim))
                - _weighted_dot_components(drag_weight_k, diff_k, lambda_drag_test, dim=int(dim))
            ) * dx
            a_drag_lambda = (
                _dot_components(d_lam_core_inv_k, lambda_drag_test, dim=int(dim))
                - _weighted_dot_components(ddrag_weight, diff_k, lambda_drag_test, dim=int(dim))
                - _weighted_dot_components(drag_weight_k, ddiff, lambda_drag_test, dim=int(dim))
            ) * dx
    # Jacobian of the vS extension penalty (k-part only).
    if float(gamma_vS_eff) != 0.0:
        if vS_ext_mode in {"l2", "mass"}:
            h_u = MeshSize()
            inv_h2 = _c(1.0) / (h_u * h_u)
            skeleton_jac_terms["extension"] = reg_weight_k * gamma_vS_c * inv_h2 * (
                (-d_alpha_support_k) * dot(vS_k, vS_test) + _one_minus(alpha_support_k) * dot(dvS, vS_test)
            ) * dx
            a_skel += skeleton_jac_terms["extension"]
        elif vS_ext_mode in {"grad", "h1"}:
            skeleton_jac_terms["extension"] = reg_weight_k * gamma_vS_c * (
                (-d_alpha_support_k) * inner(grad(vS_k), grad(vS_test))
                + _one_minus(alpha_support_k) * inner(grad(dvS), grad(vS_test))
            ) * dx
            a_skel += skeleton_jac_terms["extension"]

            if float(gamma_vS_pin_eff) != 0.0:
                h_u = MeshSize()
                inv_h2 = _c(1.0) / (h_u * h_u)
                w_pin = _one_minus(alpha_support_k)
                w_pin_pow = w_pin
                for _ in range(vS_pin_pow - 1):
                    w_pin_pow = w_pin_pow * w_pin

                if vS_pin_pow == 1:
                    w_pin_pow_m1 = _c(1.0)
                else:
                    w_pin_pow_m1 = w_pin
                    for _ in range(vS_pin_pow - 2):
                        w_pin_pow_m1 = w_pin_pow_m1 * w_pin
                dw_pin_pow = (-_c(float(vS_pin_pow)) * w_pin_pow_m1) * d_alpha_support_k
                skeleton_jac_terms["extension_pin"] = reg_weight_k * gamma_vS_pin_c * inv_h2 * (
                    dw_pin_pow * dot(vS_k, vS_test) + w_pin_pow * dot(dvS, vS_test)
                ) * dx
                a_skel += skeleton_jac_terms["extension_pin"]
        else:
            raise ValueError(f"Unknown vS_extension_mode {vS_extension_mode!r}.")

    # Optional Eulerian skeleton inertia.
    #
    # Keep the interface band purely stress-driven here as well. The support
    # fraction B belongs in the support-content transport law and in the
    # support-localized stresses, but not in inertial transfer terms: a
    # B-weighted conservative momentum transport would inject artificial
    # grad(B)-driven inertia into the diffuse interface band.
    #
    # We therefore use the intrinsic support density rho_s0_tilde in the
    # acceleration/convection terms, while B still evolves in its own
    # conservative content equation below.
    use_parametric_skeleton_inertia = any(
        weight is not None
        for weight in (
            skeleton_acceleration_weight,
            skeleton_inertia_full_weight,
            skeleton_inertia_lagged_weight,
        )
    )
    if (bool(include_skeleton_acceleration) or use_parametric_skeleton_inertia) and float(rho_s0_tilde) != 0.0:
        inertia_conv_key = str(skeleton_inertia_convection).strip().lower() if skeleton_inertia_convection is not None else "lagged"
        if inertia_conv_key in {"conservative", "nonlinear"}:
            inertia_conv_key = "full"
        if inertia_conv_key in {"picard", "semi", "semi_implicit", "linear"}:
            inertia_conv_key = "lagged"

        rho_s0_c = rho_s0_tilde

        # Keep the full one-domain internal-conversion branch consistent with
        # the reduced deformation-only model: the Eulerian skeleton inertia is
        # carried only by the support-bearing phase B = alpha * (1 - phi).
        if support_physics_key == "internal_conversion":
            rhoS_k = rho_s0_c * B_k
            rhoS_n = rho_s0_c * B_n
        else:
            rhoS_k = rho_s0_c
            rhoS_n = rho_s0_c

        if use_parametric_skeleton_inertia:
            skel_accel_w = _as_scalar_expr(
                1.0 if skeleton_acceleration_weight is None else skeleton_acceleration_weight
            )
            skel_inertia_full_w = _as_scalar_expr(
                0.0 if skeleton_inertia_full_weight is None else skeleton_inertia_full_weight
            )
            skel_inertia_lagged_w = _as_scalar_expr(
                0.0 if skeleton_inertia_lagged_weight is None else skeleton_inertia_lagged_weight
            )
        else:
            skel_accel_w = _c(1.0)
            skel_inertia_full_w = _c(1.0 if inertia_conv_key == "full" else 0.0)
            skel_inertia_lagged_w = _c(1.0 if inertia_conv_key == "lagged" else 0.0)
        skel_accel_full_w = skel_accel_w * skel_inertia_full_w
        skel_accel_lagged_w = skel_accel_w * skel_inertia_lagged_w

        # Conservative-in-time momentum term.
        momS_dot = (rhoS_k * vS_k - rhoS_n * vS_n) * inv_dt

        skeleton_terms["inertia_transient"] = skel_accel_w * inner(momS_dot, vS_test) * dx
        r_skeleton += skeleton_terms["inertia_transient"]

        # Conservative convection (two modes):
        # - full:    div(ρ_S^k v^{S,k} ⊗ v^{S,k}) (nonlinear)
        # - lagged:  div(ρ_S^n v^{S,n} ⊗ v^{S,k}) (Picard linearization)
        if inertia_conv_key not in {"full", "lagged"}:
            raise ValueError(
                f"Unknown skeleton_inertia_convection={skeleton_inertia_convection!r}. "
                "Use 'lagged' (default) or 'full'."
            )

        grad_vS_k = grad(vS_k)
        grad_vS_n = grad(vS_n)

        if support_physics_key == "internal_conversion":
            div_rhoS_vS_n = rho_s0_c * (B_n * div_vS_n + dot(gradB_n, vS_n))
        else:
            div_rhoS_vS_n = rho_s0_c * div_vS_n
        advS_full_k = dot(grad_vS_k, vS_k)
        advS_full_n = dot(grad_vS_n, vS_n)
        convS_full_k = dot(advS_full_k, vS_test)
        convS_full_n = dot(advS_full_n, vS_test)
        if support_physics_key == "internal_conversion":
            div_rhoS_vS_k = rho_s0_c * (B_k * div_vS_k + dot(gradB_k, vS_k))
        else:
            div_rhoS_vS_k = rho_s0_c * div_vS_k
        # Lagged/Picard form: div(ρ^n v^n ⊗ v^k) = ρ^n (v^n·∇)v^k + v^k div(ρ^n v^n).
        advS_lagged_k = dot(grad_vS_k, vS_n)
        convS_lagged_k = dot(advS_lagged_k, vS_test)
        advS_lagged_n = dot(grad_vS_n, vS_n)
        convS_lagged_n = dot(advS_lagged_n, vS_test)

        skeleton_terms["inertia_convection_k"] = (
            skel_accel_full_w * th * (rhoS_k * convS_full_k + div_rhoS_vS_k * dot(vS_k, vS_test)) * dx
            + skel_accel_lagged_w * th * (rhoS_n * convS_lagged_k + div_rhoS_vS_n * dot(vS_k, vS_test)) * dx
        )
        skeleton_terms["inertia_convection_n"] = (
            skel_accel_full_w * one_m_th * (rhoS_n * convS_full_n + div_rhoS_vS_n * dot(vS_n, vS_test)) * dx
            + skel_accel_lagged_w * one_m_th * (rhoS_n * convS_lagged_n + div_rhoS_vS_n * dot(vS_n, vS_test)) * dx
        )
        r_skeleton += skeleton_terms["inertia_convection_k"]
        r_skeleton += skeleton_terms["inertia_convection_n"]

        # Jacobian (k-part only): always include δ[ (ρ_S v^S)/dt ].
        if support_physics_key == "internal_conversion":
            d_rhoS_k = rho_s0_c * dB_k
        else:
            d_rhoS_k = _c(0.0)
        d_momS_dot_vtest = _c(0.0)
        for i in range(int(dim)):
            d_momS_dot_vtest += (
                _vector_component(vS_k, i) * d_rhoS_k + rhoS_k * _vector_component(dvS, i)
            ) * _vector_component(vS_test, i)
        skeleton_jac_terms["inertia_transient"] = skel_accel_w * inv_dt * d_momS_dot_vtest * dx
        a_skel += skeleton_jac_terms["inertia_transient"]

        grad_dvS = grad(dvS)
        d_advS_full_k = dot(grad_dvS, vS_k) + dot(grad_vS_k, dvS)
        d_convS_full_k = dot(d_advS_full_k, vS_test)
        if support_physics_key == "internal_conversion":
            d_div_rhoS_vS_k = rho_s0_c * (
                dB_k * div_vS_k
                + B_k * div_dvS
                + dot(dgradB_k, vS_k)
                + dot(gradB_k, dvS)
            )
        else:
            d_div_rhoS_vS_k = rho_s0_c * div_dvS
        d_advS_lagged_k = dot(grad_dvS, vS_n)
        d_convS_lagged_k = dot(d_advS_lagged_k, vS_test)
        skeleton_jac_terms["inertia_convection_density"] = (
            skel_accel_full_w * th * (convS_full_k * d_rhoS_k + rhoS_k * d_convS_full_k) * dx
            + skel_accel_lagged_w * th * (rhoS_n * d_convS_lagged_k) * dx
        )
        a_skel += skeleton_jac_terms["inertia_convection_density"]

        skeleton_jac_terms["inertia_convection_divergence"] = (
            skel_accel_full_w * th * (
                dot(vS_k, vS_test) * d_div_rhoS_vS_k + div_rhoS_vS_k * dot(dvS, vS_test)
            ) * dx
            + skel_accel_lagged_w * th * (div_rhoS_vS_n * dot(dvS, vS_test)) * dx
        )
        a_skel += skeleton_jac_terms["inertia_convection_divergence"]

    # NOTE: u-CIP stabilization is applied to the *kinematic constraint* for u
    # (see below), not to the vS momentum balance.
    # ------------------------------------------------------------------
    # (iii-b) Optional wall adhesion traction (spring + dashpot)
    # ------------------------------------------------------------------
    if ds_adh is not None and (
        float(adhesion_k_n) != 0.0
        or float(adhesion_k_t) != 0.0
        or float(adhesion_gamma_n) != 0.0
        or float(adhesion_gamma_t) != 0.0
    ):
        # Adhesion integrity a is treated as lagged/known in the Newton step.
        a_prev = adhesion_a_prev if adhesion_a_prev is not None else _c(1.0)

        k_n_c = _c(float(adhesion_k_n))
        k_t_c = _c(float(adhesion_k_t))
        g_n_c = _c(float(adhesion_gamma_n))
        g_t_c = _c(float(adhesion_gamma_t))

        n_b = FacetNormal()

        def _proj_n(vec):
            return dot(vec, n_b) * n_b

        def _proj_t(vec):
            return vec - _proj_n(vec)

        # k-level traction uses (u_k, vS_k); n-level uses (u_n, vS_n) for θ-scheme consistency.
        u_nvec_k = _proj_n(u_k)
        u_tvec_k = u_k - u_nvec_k
        vS_nvec_k = _proj_n(vS_k)
        vS_tvec_k = vS_k - vS_nvec_k

        u_nvec_n = _proj_n(u_n)
        u_tvec_n = u_n - u_nvec_n
        vS_nvec_n = _proj_n(vS_n)
        vS_tvec_n = vS_n - vS_nvec_n

        t_adh_k = k_n_c * u_nvec_k + k_t_c * u_tvec_k + g_n_c * vS_nvec_k + g_t_c * vS_tvec_k
        t_adh_n = k_n_c * u_nvec_n + k_t_c * u_tvec_n + g_n_c * vS_nvec_n + g_t_c * vS_tvec_n

        skeleton_terms["adhesion"] = (
            sk_th * alpha_support_k * a_prev * dot(t_adh_k, vS_test)
            + sk_one_m_th * alpha_support_n * a_prev * dot(t_adh_n, vS_test)
        ) * ds_adh
        r_skeleton += skeleton_terms["adhesion"]

        # Jacobian (k-part only): δ[ α a_prev t_adh(u,vS) ].
        du_nvec = _proj_n(du)
        du_tvec = du - du_nvec
        dvS_nvec = _proj_n(dvS)
        dvS_tvec = dvS - dvS_nvec
        dt_adh = k_n_c * du_nvec + k_t_c * du_tvec + g_n_c * dvS_nvec + g_t_c * dvS_tvec

        skeleton_jac_terms["adhesion"] = sk_th * (
            d_alpha_support_k * a_prev * dot(t_adh_k, vS_test) + alpha_support_k * a_prev * dot(dt_adh, vS_test)
        ) * ds_adh
        a_skel += skeleton_jac_terms["adhesion"]

    # Optional CIP (continuous interior penalty) stabilization for vS.
    # Regularizes vS in the diffuse interface / near-zero-α region without changing the continuous limit.
    if float(vS_cip) != 0.0 and ds_cip is not None:
        vS_cip_c = _as_constant(vS_cip)
        n_int = FacetNormal()
        h_F = avg(MeshSize())
        tau_cip = vS_cip_c * (h_F * h_F * h_F) * inv_dt
        w_s = avg(alpha_support_n)
        skeleton_terms["cip"] = tau_cip * reg_weight_cip * w_s * _grad_inner_jump(vS_k, vS_test, n_int) * ds_cip
        r_skeleton += skeleton_terms["cip"]
        skeleton_jac_terms["cip"] = tau_cip * reg_weight_cip * w_s * _grad_inner_jump(dvS, vS_test, n_int) * ds_cip
        a_skel += skeleton_jac_terms["cip"]

    # ------------------------------------------------------------------
    # (iii-c) Solid kinematics (Eulerian reference-map constraint)
    # ------------------------------------------------------------------
    stored_support_vS_gate_n = _c(1.0)
    grad_stored_support_vS_gate_n = zero_vector
    if support_physics_key == "stored_support":
        gate_alpha0 = float(alpha_vS_gate_alpha0)
        if gate_alpha0 > 0.0:
            gate_pow = int(alpha_vS_gate_power)
            if gate_pow < 1:
                raise ValueError(f"alpha_vS_gate_power must be >= 1; got {alpha_vS_gate_power}.")
            gate_num = alpha_n
            for _ in range(gate_pow - 1):
                gate_num = gate_num * alpha_n
            gate_denom = gate_num + _c(gate_alpha0 ** float(gate_pow)) + _c(1.0e-12)
            stored_support_vS_gate_n = gate_num / gate_denom
            grad_stored_support_vS_gate_n = grad(stored_support_vS_gate_n)

    vS_kin_k = stored_support_vS_gate_n * vS_k
    vS_kin_n = stored_support_vS_gate_n * vS_n
    dvS_kin = stored_support_vS_gate_n * dvS

    # For an Eulerian reference map X(x,t) (material coordinate of the point at x),
    #   ∂_t X + vS·∇X = 0.
    # With u = x - X, this becomes:
    #   ∂_t u + vS·∇u = vS.
    #
    # We enforce this as a separate (first-order) PDE for u, localized to the
    # biofilm region via α. Outside the biofilm (α≈0), u is defined by the
    # extension penalty below.
    #
    # Scaling: multiplying the *entire* u-equation by a positive scalar does not
    # change the solution set, but it can improve conditioning of the monolithic
    # Newton solve and the line-search norm weighting (important when the u
    # residual is orders of magnitude smaller than the vS residual).
    if kinematics_scale is None:
        kinematics_scale = rho_s0_tilde if (rho_s0_tilde is not None and float(rho_s0_tilde) != 0.0) else 1.0
    kin_scale_c = kinematics_scale if hasattr(kinematics_scale, "dim") else _c(float(kinematics_scale))

    Fkin_dt = (u_k - u_n) * inv_dt
    Fkin_adv_k = dot(grad(u_k), vS_kin_k) - vS_kin_k
    Fkin_adv_n = dot(grad(u_n), vS_kin_n) - vS_kin_n
    Fkin_k = Fkin_dt + th * Fkin_adv_k + one_m_th * Fkin_adv_n

    kinematics_zero = kin_scale_c * _c(0.0) * dot(u_k, u_test) * dx
    kinematics_terms: dict[str, object] = {
        "base": kin_scale_c * alpha_support_k * dot(Fkin_k, u_test) * dx,
        "supg": kinematics_zero,
        "extension": kinematics_zero,
        "cip": kinematics_zero,
    }
    r_kinematics = kinematics_terms["base"]

    # Jacobian (k-part only)
    dFkin_dt = du * inv_dt
    dFkin_adv_k = dot(grad(du), vS_kin_k) + dot(grad(u_k), dvS_kin) - dvS_kin
    dFkin_k = dFkin_dt + th * dFkin_adv_k
    a_kinematics = kin_scale_c * (d_alpha_support_k * dot(Fkin_k, u_test) + alpha_support_k * dot(dFkin_k, u_test)) * dx

    # Optional SUPG-like streamline diffusion for the u-transport (kinematic constraint).
    #
    # This adds an artificial diffusion along vS in the solid region:
    #   τ ( (vS^n·∇)u^k , (vS^n·∇)ξ )_{Ω}  localized by α^n.
    if float(u_supg) != 0.0:
        u_supg_c = _as_constant(u_supg)
        h_u = MeshSize()
        vmag2 = vS_kin_n[0] * vS_kin_n[0] + vS_kin_n[1] * vS_kin_n[1]
        vmag = _sqrt(vmag2 + _c(1.0e-12))
        denom = h_u * vmag + (h_u * h_u) * inv_dt
        tau_u = u_supg_c * (h_u * h_u) / (denom + _c(1.0e-16))
        w_u = alpha_support_n  # lagged "solid-only" localization
        adv_u_k = dot(grad(u_k), vS_kin_n)
        adv_xi = dot(grad(u_test), vS_kin_n)
        kinematics_terms["supg"] = kin_scale_c * tau_u * w_u * inner(adv_u_k, adv_xi) * dx
        r_kinematics += kinematics_terms["supg"]
        a_kinematics += kin_scale_c * tau_u * w_u * inner(dot(grad(du), vS_kin_n), adv_xi) * dx

    # Optional extension penalty to keep u well-posed in the free-fluid region (α≈0).
    if float(gamma_u) != 0.0:
        u_ext_mode = str(u_extension_mode).strip().lower()
        if u_ext_mode in {"l2", "mass"}:
            h_u = MeshSize()
            inv_h2 = _c(1.0) / (h_u * h_u)
            extension_term = kin_scale_c * reg_weight_k * gamma_u_c * inv_h2 * _one_minus(alpha_support_k) * dot(u_k, u_test) * dx
            kinematics_terms["extension"] = kinematics_terms["extension"] + extension_term
            r_kinematics += extension_term
            a_kinematics += kin_scale_c * reg_weight_k * gamma_u_c * inv_h2 * (
                (-d_alpha_support_k) * dot(u_k, u_test) + _one_minus(alpha_support_k) * dot(du, u_test)
            ) * dx
        elif u_ext_mode in {"grad", "h1"}:
            extension_term = kin_scale_c * reg_weight_k * gamma_u_c * _one_minus(alpha_support_k) * inner(grad(u_k), grad(u_test)) * dx
            kinematics_terms["extension"] = kinematics_terms["extension"] + extension_term
            r_kinematics += extension_term
            a_kinematics += kin_scale_c * reg_weight_k * gamma_u_c * (
                (-d_alpha_support_k) * inner(grad(u_k), grad(u_test)) + _one_minus(alpha_support_k) * inner(grad(du), grad(u_test))
            ) * dx

            if float(gamma_u_pin) != 0.0:
                h_u = MeshSize()
                inv_h2 = _c(1.0) / (h_u * h_u)
                pin_c = _c(float(gamma_u_pin))
                w_pin = _one_minus(alpha_support_k)
                w_pin2 = w_pin * w_pin
                dw_pin2 = (-_c(2.0) * w_pin) * d_alpha_support_k
                pin_term = kin_scale_c * reg_weight_k * pin_c * inv_h2 * w_pin2 * dot(u_k, u_test) * dx
                kinematics_terms["extension"] = kinematics_terms["extension"] + pin_term
                r_kinematics += pin_term
                a_kinematics += kin_scale_c * reg_weight_k * pin_c * inv_h2 * (dw_pin2 * dot(u_k, u_test) + w_pin2 * dot(du, u_test)) * dx
        else:
            raise ValueError(f"Unknown u_extension_mode {u_extension_mode!r}.")

    # Optional facet stabilization for u (CIP/ghost-penalty on interior facets).
    if float(u_cip) != 0.0 and ds_cip is not None:
        n_int = FacetNormal()
        h_F = avg(MeshSize())
        # NOTE: keep the CIP weight backend-identical. A velocity-dependent term
        # (|vS|/h) would require a robust sqrt on interior facets; use inv_dt.
        scale = inv_dt
        tau_u_cip = _c(float(u_cip)) * (h_F * h_F * h_F) * scale
        w_key = str(u_cip_weight or "fluid").strip().lower()
        if w_key in {"fluid", "one_minus_alpha", "1-alpha", "one-minus-alpha"}:
            w_u_cip = avg(_one_minus(alpha_n))
        elif w_key in {"biofilm", "alpha"}:
            w_u_cip = avg(alpha_n)
        elif w_key in {"both", "all", "unity", "1"}:
            w_u_cip = _c(1.0)
        else:
            raise ValueError(
                f"Unknown u_cip_weight={u_cip_weight!r}. Use 'fluid' (default), 'biofilm', or 'both'."
            )
        kinematics_terms["cip"] = kin_scale_c * tau_u_cip * reg_weight_cip * w_u_cip * _grad_inner_jump(u_k, u_test, n_int) * ds_cip
        r_kinematics += kinematics_terms["cip"]
        a_kinematics += kin_scale_c * tau_u_cip * reg_weight_cip * w_u_cip * _grad_inner_jump(du, u_test, n_int) * ds_cip

    # ------------------------------------------------------------------
    # (iv) Porosity / solid-content transport
    # ------------------------------------------------------------------
    D_phi_c = _as_constant(D_phi)
    gamma_phi_c = _c(float(gamma_phi))
    one_m_alpha_k = _one_minus(alpha_k)
    w_phi_fluid4_k = one_m_alpha_k * one_m_alpha_k
    w_phi_fluid4_k = w_phi_fluid4_k * w_phi_fluid4_k
    w_phi_fluid8_k = w_phi_fluid4_k * w_phi_fluid4_k
    w_phi_fluid_k = w_phi_fluid8_k * w_phi_fluid8_k
    one_m_alpha3_k = one_m_alpha_k * one_m_alpha_k * one_m_alpha_k
    dw_phi_fluid_k = (-_c(16.0) * (w_phi_fluid8_k * w_phi_fluid4_k * one_m_alpha3_k)) * dalpha
    phi_diff_weight_key = str(phi_diffusion_weight or "fluid").strip().lower()
    if phi_diff_weight_key in {"unity", "1", "all"}:
        w_phi_diff_k = _c(1.0)
        dw_phi_diff_k = _c(0.0)
    elif phi_diff_weight_key in {"fluid", "one_minus_alpha", "1-alpha", "one-minus-alpha"}:
        w_phi_diff_k = w_phi_fluid_k
        dw_phi_diff_k = dw_phi_fluid_k
    elif phi_diff_weight_key in {"biofilm", "alpha"}:
        w_phi_diff_k = alpha_k
        dw_phi_diff_k = dalpha
    else:
        raise ValueError(
            f"Unknown phi_diffusion_weight={phi_diffusion_weight!r}. "
            "Use 'unity', 'fluid', or 'biofilm'."
        )

    mu_max_c = _c(float(mu_max))
    K_S_c = _c(float(K_S))
    mon_k = _monod(S_k, mu_max=mu_max_c, K_S=K_S_c)
    dmon_dS = mu_max_c * (K_S_c / ((S_k + K_S_c) * (S_k + K_S_c)))
    # The conservative solid-volume growth source is defined through B in both
    # the ratio-free (B-primary) and legacy phi-based branches.
    Pi_k = _Pi_over_rho_s_B(S_k, B_k, mu_max=mu_max_c, K_S=K_S_c, k_d=_c(float(k_d)))
    Pi_n = _Pi_over_rho_s_B(S_n, B_n, mu_max=mu_max_c, K_S=K_S_c, k_d=_c(float(k_d)))
    dPi = (dmon_dS * dS) * B_k + (mon_k - _c(float(k_d))) * dB_k
    if use_B_primary:
        r_phi = _c(0.0) * alpha_test * dx
        a_phi = _c(0.0) * dalpha * alpha_test * dx
        if support_physics_key == "stored_support" and stored_support_content_key == "freeze_b":
            r_B = B_test * ((B_k - B_n) * inv_dt) * dx
            a_B = B_test * (dB * inv_dt) * dx
        elif support_physics_key == "stored_support" and stored_support_content_key == "frozen_phi_b":
            frozen_B_k = (_c(1.0) - _as_constant(phi_b)) * alpha_k
            frozen_dB_k = (_c(1.0) - _as_constant(phi_b)) * dalpha
            r_B = B_test * (B_k - frozen_B_k) * dx
            a_B = B_test * (dB - frozen_dB_k) * dx
        else:
            flux_B_k = _c(0.0)
            flux_B_n = _c(0.0)
            dflux_B_k = _c(0.0)
            for i in range(int(dim)):
                grad_i = grad(B_test)[i]
                flux_B_k += (B_k * _vector_component(vS_k, i)) * grad_i
                flux_B_n += (B_n * _vector_component(vS_n, i)) * grad_i
                dflux_B_k += (dB * _vector_component(vS_k, i) + B_k * _vector_component(dvS, i)) * grad_i
            r_B = B_test * ((B_k - B_n) * inv_dt) * dx
            r_B += -th * flux_B_k * dx
            r_B += -one_m_th * flux_B_n * dx
            if ds_B_transport is not None:
                n_b = FacetNormal()
                flux_B_bdry_k = _c(0.0)
                flux_B_bdry_n = _c(0.0)
                dflux_B_bdry_k = _c(0.0)
                for i in range(int(dim)):
                    n_i = n_b[i]
                    flux_B_bdry_k += B_k * _vector_component(vS_k, i) * n_i
                    flux_B_bdry_n += B_n * _vector_component(vS_n, i) * n_i
                    dflux_B_bdry_k += (dB * _vector_component(vS_k, i) + B_k * _vector_component(dvS, i)) * n_i
                r_B += th * B_test * flux_B_bdry_k * ds_B_transport
                r_B += one_m_th * B_test * flux_B_bdry_n * ds_B_transport
            r_B += -B_test * (th * Pi_k + one_m_th * Pi_n) * dx
            r_B += -B_test * f_phi * dx

            a_B = B_test * (dB * inv_dt) * dx
            a_B += -th * dflux_B_k * dx
            if ds_B_transport is not None:
                a_B += th * B_test * dflux_B_bdry_k * ds_B_transport
            a_B += -th * B_test * dPi * dx
    elif support_physics_key == "internal_conversion":
        # Conservative solid-volume balance for B = alpha (1-phi):
        #
        #   ∂t B + div(B vS) = Γ,   Γ = Π_b / ρ_s*
        #
        # Growth therefore appears once in the whole-domain solid-volume
        # equation, and alpha itself stays source-free.
        #
        # The gamma_phi term kept below is *not* part of the physical B-balance.
        # It acts only as an outside-support gauge for the underdetermined free-fluid
        # phi extension, localized by (1-alpha)^16 so it does not contaminate the
        # support interior in the sharp-interface limit.
        flux_dot_grad_test_k = _c(0.0)
        flux_dot_grad_test_n = _c(0.0)
        dflux_dot_grad_test_k = _c(0.0)
        for i in range(int(dim)):
            flux_dot_grad_test_k = flux_dot_grad_test_k + (B_k * _vector_component(vS_k, i)) * grad(phi_test)[i]
            flux_dot_grad_test_n = flux_dot_grad_test_n + (B_n * _vector_component(vS_n, i)) * grad(phi_test)[i]
            dflux_dot_grad_test_k = dflux_dot_grad_test_k + (
                (dB_k * _vector_component(vS_k, i) + B_k * _vector_component(dvS, i)) * grad(phi_test)[i]
            )

        r_phi = phi_test * ((B_k - B_n) * inv_dt) * dx
        r_phi += -th * flux_dot_grad_test_k * dx
        r_phi += -one_m_th * flux_dot_grad_test_n * dx
        if ds_B_transport is not None:
            n_b = FacetNormal()
            flux_B_bdry_k = _c(0.0)
            flux_B_bdry_n = _c(0.0)
            dflux_B_bdry_k = _c(0.0)
            for i in range(int(dim)):
                n_i = _vector_component(n_b, i)
                flux_B_bdry_k += B_k * _vector_component(vS_k, i) * n_i
                flux_B_bdry_n += B_n * _vector_component(vS_n, i) * n_i
                dflux_B_bdry_k += (dB_k * _vector_component(vS_k, i) + B_k * _vector_component(dvS, i)) * n_i
            r_phi += th * phi_test * flux_B_bdry_k * ds_B_transport
            r_phi += one_m_th * phi_test * flux_B_bdry_n * ds_B_transport
        r_phi += -phi_test * (th * Pi_k + one_m_th * Pi_n) * dx
        r_phi += D_phi_c * w_phi_diff_k * inner(grad(phi_k), grad(phi_test)) * dx
        r_phi += gamma_phi_c * w_phi_fluid_k * (phi_k - _c(1.0)) * phi_test * dx
        r_phi += -phi_test * f_phi * dx

        a_phi = phi_test * (dB_k * inv_dt) * dx
        a_phi += -th * dflux_dot_grad_test_k * dx
        if ds_B_transport is not None:
            a_phi += th * phi_test * dflux_B_bdry_k * ds_B_transport
        a_phi += -th * phi_test * dPi * dx
        a_phi += D_phi_c * (
            dw_phi_diff_k * inner(grad(phi_k), grad(phi_test))
            + w_phi_diff_k * inner(grad(dphi), grad(phi_test))
        ) * dx
        a_phi += gamma_phi_c * (dw_phi_fluid_k * (phi_k - _c(1.0)) + w_phi_fluid_k * dphi) * phi_test * dx

        # Optional stabilization for the conservative B-balance.
        #
        # We stabilize the transported solid-volume variable
        #
        #   B = alpha (1-phi),
        #
        # not the legacy inner porosity law. The SUPG term uses the strong
        # residual of
        #
        #   ∂t B + div(B vS) - Γ = 0,
        #
        # with lagged streamline direction vS^n and lagged support localization
        # alpha^n so the added term remains consistent and focused on the support
        # transport rather than the free-fluid gauge extension.
        if float(phi_supg) != 0.0:
            h_p = MeshSize()
            vmag2 = _c(0.0)
            for i in range(int(dim)):
                vmag2 += _vector_component(vS_n, i) * _vector_component(vS_n, i)
            vmag = _sqrt(vmag2 + _c(1.0e-12))
            denom = (_c(2.0) * inv_dt) * (_c(2.0) * inv_dt) + (_c(2.0) * vmag / (h_p + _c(1.0e-12))) * (
                _c(2.0) * vmag / (h_p + _c(1.0e-12))
            )
            tau_supg = _c(float(phi_supg)) / _sqrt(denom + _c(1.0e-16))
            g_test = grad(phi_test)
            w_supg = _c(0.0)
            for i in range(int(dim)):
                w_supg += alpha_n * g_test[i] * _vector_component(vS_n, i)

            f_phi_supg_k = (B_k - B_n) * inv_dt
            f_phi_supg_k += th * divBvS_k + one_m_th * divBvS_n
            f_phi_supg_k += -(th * Pi_k + one_m_th * Pi_n)
            f_phi_supg_k += gamma_phi_c * w_phi_fluid_k * (phi_k - _c(1.0))
            f_phi_supg_k += -f_phi

            r_phi += tau_supg * w_supg * f_phi_supg_k * dx

            df_phi_supg_k = dB_k * inv_dt
            df_phi_supg_k += th * d_divBvS_k
            df_phi_supg_k += -th * dPi
            df_phi_supg_k += gamma_phi_c * (dw_phi_fluid_k * (phi_k - _c(1.0)) + w_phi_fluid_k * dphi)

            a_phi += tau_supg * w_supg * df_phi_supg_k * dx

        if float(phi_cip) != 0.0 and ds_cip is not None:
            n_int = FacetNormal()
            h_F = avg(MeshSize())
            scale = inv_dt
            tau_cip = _c(float(phi_cip)) * (h_F * h_F * h_F) * scale
            # Localize the B-CIP term to the transported support. The transported
            # quantity is B, so the penalty acts on grad(B), not grad(phi).
            a_avg = avg(alpha_n)
            a_jump = jump(alpha_n)
            w_phi_cip = a_avg * a_avg + (-_c(0.25) * a_jump * a_jump)
            # For CG alpha/phi, the facet traces of alpha and phi are continuous,
            # so the normal jump of grad(B) = grad(alpha (1-phi)) can be expanded
            # exactly as
            #
            #   jump(grad(B), n)
            #     = (1-avg(phi)) jump(grad(alpha), n) - avg(alpha) jump(grad(phi), n).
            #
            # This avoids backend-sensitive grad(product) paths on interior facets
            # while remaining mathematically identical for the CG support fields.
            phi_trace_k = avg(phi_k)
            alpha_trace_k = avg(alpha_k)
            j_alpha_k = jump(grad(alpha_k), n_int)
            j_phi_k = jump(grad(phi_k), n_int)
            j_B_k = _one_minus(phi_trace_k) * j_alpha_k - alpha_trace_k * j_phi_k
            j_phi_test = jump(grad(phi_test), n_int)
            r_phi += tau_cip * w_phi_cip * inner(j_B_k, j_phi_test) * ds_cip
            j_dalpha = jump(grad(dalpha), n_int)
            j_dphi = jump(grad(dphi), n_int)
            dj_B_k = _one_minus(phi_trace_k) * j_dalpha - alpha_trace_k * j_dphi
            a_phi += tau_cip * w_phi_cip * inner(dj_B_k, j_phi_test) * ds_cip
        r_B = r_phi
        a_B = a_phi
    else:
        # Legacy inner-region porosity equation:
        #   ∂t(α φ) + α [ vS·∇φ - (1-φ) div(vS) + Π/ρ_s* ] = ...
        # This is kept for backward compatibility with the historical support
        # creation/destruction model.
        Fphi_k = dot(grad(phi_k), vS_k) - _one_minus(phi_k) * div_vS_k + Pi_k
        Fphi_n = dot(grad(phi_n), vS_n) - _one_minus(phi_n) * div_vS_n + Pi_n

        r_phi = alpha_k * phi_test * ((phi_k - phi_n) * inv_dt) * dx
        r_phi += th * alpha_k * phi_test * Fphi_k * dx
        r_phi += one_m_th * alpha_n * phi_test * Fphi_n * dx
        r_phi += D_phi_c * w_phi_diff_k * inner(grad(phi_k), grad(phi_test)) * dx
        r_phi += gamma_phi_c * w_phi_fluid_k * (phi_k - _c(1.0)) * phi_test * dx
        r_phi += -phi_test * f_phi * dx

        a_phi = alpha_k * phi_test * (dphi * inv_dt) * dx
        a_phi += dalpha * phi_test * ((phi_k - phi_n) * inv_dt) * dx
        a_phi += th * dalpha * phi_test * Fphi_k * dx
        a_phi += th * alpha_k * phi_test * (
            dot(grad(phi_k), dvS) + dot(grad(dphi), vS_k) + dphi * div_vS_k - _one_minus(phi_k) * div_dvS + dPi
        ) * dx
        a_phi += D_phi_c * (
            dw_phi_diff_k * inner(grad(phi_k), grad(phi_test))
            + w_phi_diff_k * inner(grad(dphi), grad(phi_test))
        ) * dx
        a_phi += gamma_phi_c * (dw_phi_fluid_k * (phi_k - _c(1.0)) + w_phi_fluid_k * dphi) * phi_test * dx

        # Optional consistent stabilization for advection-dominated φ (useful with D_phi=0).
        # - SUPG: τ (vS·∇w) R_phi
        # - CIP:  γ h^3 (1/dt + |vS|/h) <[∂_n φ],[∂_n w]>_F on interior facets
        #
        # We again use lagged vS_n in τ and in the test-direction to keep the Jacobian
        # coupling limited to the (already-present) vS_k dependence inside R_phi.
        if float(phi_supg) != 0.0:
            h_p = MeshSize()
            vmag2 = vS_n[0] * vS_n[0] + vS_n[1] * vS_n[1]
            vmag = _sqrt(vmag2 + _c(1.0e-12))
            denom = (_c(2.0) * inv_dt) * (_c(2.0) * inv_dt) + (_c(2.0) * vmag / (h_p + _c(1.0e-12))) * (
                _c(2.0) * vmag / (h_p + _c(1.0e-12))
            )
            tau_supg = _c(float(phi_supg)) / _sqrt(denom + _c(1.0e-16))
            # Avoid `dot(grad(test), v)` due to current dot/left_dot limitations in the
            # C++ backend for some mixed expression types; expand componentwise.
            g_test = grad(phi_test)
            w_supg = g_test[0] * vS_n[0] + g_test[1] * vS_n[1]

            # "Strong" residual (excluding diffusion; diffusion is a stabilizing term already).
            f_phi_supg_k = alpha_k * ((phi_k - phi_n) * inv_dt)
            f_phi_supg_k += th * alpha_k * Fphi_k + one_m_th * alpha_n * Fphi_n
            f_phi_supg_k += gamma_phi_c * w_phi_fluid_k * (phi_k - _c(1.0))
            f_phi_supg_k += -f_phi

            r_phi += tau_supg * w_supg * f_phi_supg_k * dx

            dFphi_k = dot(grad(phi_k), dvS) + dot(grad(dphi), vS_k) + dphi * div_vS_k - _one_minus(phi_k) * div_dvS + dPi
            df_phi_supg_k = dalpha * ((phi_k - phi_n) * inv_dt) + alpha_k * (dphi * inv_dt)
            df_phi_supg_k += th * (dalpha * Fphi_k + alpha_k * dFphi_k)
            df_phi_supg_k += gamma_phi_c * (dw_phi_fluid_k * (phi_k - _c(1.0)) + w_phi_fluid_k * dphi)

            a_phi += tau_supg * w_supg * df_phi_supg_k * dx

        if float(phi_cip) != 0.0 and ds_cip is not None:
            n_int = FacetNormal()
            h_F = avg(MeshSize())
            scale = inv_dt
            tau_cip = _c(float(phi_cip)) * (h_F * h_F * h_F) * scale
            # Localize facet stabilization to the biofilm region to avoid smearing the
            # imposed fluid value (φ=1) into the biofilm over long time horizons.
            #
            # Use α^- α^+ (computed robustly from avg/jump) as a weight:
            #   α^- α^+ = avg(α)^2 - (jump(α)/2)^2
            a_avg = avg(alpha_n)
            a_jump = jump(alpha_n)
            w_phi_cip = a_avg * a_avg + (-_c(0.25) * a_jump * a_jump)
            r_phi += tau_cip * w_phi_cip * _grad_inner_jump(phi_k, phi_test, n_int) * ds_cip
            a_phi += tau_cip * w_phi_cip * _grad_inner_jump(dphi, phi_test, n_int) * ds_cip
        r_B = None
        a_B = None

    # ------------------------------------------------------------------
    # (v) Indicator evolution (advection–diffusion–reaction)
    # ------------------------------------------------------------------
    D_alpha_c = _as_constant(D_alpha)
    k_g_c = k_g if hasattr(k_g, "dim") else _c(float(k_g))
    mu_max_c = mu_max if hasattr(mu_max, "dim") else _c(float(mu_max))
    K_S_c = K_S if hasattr(K_S, "dim") else _c(float(K_S))
    if use_B_primary:
        G_k = zero_scalar
        G_n = zero_scalar
    else:
        G_k = _G(S_k, phi_k, k_g=k_g_c, mu_max=mu_max_c, K_S=K_S_c)
        G_n = _G(S_n, phi_n, k_g=k_g_c, mu_max=mu_max_c, K_S=K_S_c)

    # Phase-field (Allen–Cahn) regularization parameters for α.
    # Only active when alpha_cahn_M*alpha_cahn_gamma != 0.
    eps_alpha_val = float(alpha_cahn_eps)
    if eps_alpha_val <= 0.0 and (float(alpha_cahn_M) != 0.0 or float(alpha_crack_k) != 0.0):
        raise ValueError(f"alpha_cahn_eps must be > 0 when phase-field/crack terms are enabled; got {eps_alpha_val}.")
    eps_alpha_c = _c(max(eps_alpha_val, 1.0e-12))

    M_alpha_c = _c(float(alpha_cahn_M))
    gamma_alpha_c = _c(float(alpha_cahn_gamma))
    M_gamma_alpha = M_alpha_c * gamma_alpha_c

    # Surface-localized detachment sink: D_det_prev * δ(α), where δ is a smooth interface delta.
    # We use δ(α) = 4 α (1-α), which is supported robustly by the current UFL/compiler stack.
    delta_k = _c(4.0) * alpha_k * _one_minus(alpha_k)
    delta_n = _c(4.0) * alpha_n * _one_minus(alpha_n)

    # Optional crack-propagation surface speed V_crack^prev (lagged): adds an additional
    # surface-localized sink proportional to δ(α).
    crack_coef_prev = _c(0.0)
    if float(alpha_crack_k) != 0.0:
        if V_crack_prev is not None:
            Vc_prev = V_crack_prev if hasattr(V_crack_prev, "dim") else _c(float(V_crack_prev))
        else:
            driver_key = str(alpha_crack_driver).strip().lower()
            eta_mech = _c(float(alpha_crack_eta_mech))

            # IMPORTANT: avoid vector inner products of function-function in LHS assembly
            # (the current visitor only supports VecOpInfo.inner for test/trial pairs).
            # Use driver measures built from GradOpInfo/HessOpInfo or scalar expressions.
            if driver_key in {"shear", "fluid_shear", "tau"}:
                # Fluid shear stress proxy (works robustly in all backends):
                #   τ ≈ 2 μ ||ε(v)||_F
                tau2 = inner(_epsilon(v_n), _epsilon(v_n))
                tau = _sqrt(tau2 + eta_mech)
                D_mech_n = _c(2.0) * mu_n * tau
            elif driver_key in {"solid_strain", "strain"}:
                # Skeleton strain proxy (dimensionless):
                #   ||ε(u)||_F
                eps2 = inner(_epsilon(u_n), _epsilon(u_n))
                D_mech_n = _sqrt(eps2 + eta_mech)
            elif driver_key in {"solid_von_mises", "von_mises", "von-mises", "vm", "solid_vm"}:
                # Solid von Mises equivalent stress (Pa), based on the *elastic* Cauchy stress.
                # Use the same constitutive law as the skeleton equation.
                if solid_model_key in {"linear", "small_strain", "linear_elastic"}:
                    sig = _c(2.0) * mu_s * _epsilon(u_n) + lambda_s * div(u_n) * Identity(int(dim))
                elif solid_model_key in {"hencky", "hencky_log", "hencky_log_strain"}:
                    sig = sigma_hencky(u_n, mu_s, lambda_s, dim=int(dim))
                elif solid_model_key in {"stvk", "svk", "saint_venant_kirchhoff", "saint-venant-kirchhoff"}:
                    sig = sigma_svk(u_n, mu_s, lambda_s, dim=int(dim))
                elif _is_seboldt_neo_hookean_model(solid_model_key):
                    sig = sigma_neo_hookean_seboldt(u_n, mu_s, lambda_s, dim=int(dim))
                else:
                    if c_nh is None:
                        c_nh = mu_s / _c(2.0)
                    if beta_nh is None:
                        beta_nh = lambda_s / (_c(2.0) * mu_s)
                    sig = sigma_neo_hookean(u_n, c_nh, beta_nh, dim=int(dim))

                tr_sig = trace(sig)
                I = Identity(int(dim))
                s_dev = sig - (tr_sig / _c(float(dim))) * I
                # σ_vm = sqrt(3/2 s:s). For 2D we still use the 3D-style J2 scaling as a robust proxy.
                vm2 = (_c(1.5) * inner(s_dev, s_dev)) + eta_mech
                D_mech_n = _sqrt(vm2)
            elif driver_key in {"drag", "brinkman_drag", "brinkman"}:
                # Approximate drag-driven tearing with the shear proxy due to current compiler limits.
                tau2 = inner(_epsilon(v_n), _epsilon(v_n))
                tau = _sqrt(tau2 + eta_mech)
                D_mech_n = _c(2.0) * mu_n * tau
            else:
                raise ValueError(
                    f"Unknown alpha_crack_driver {alpha_crack_driver!r}. Use 'shear', 'solid_strain', 'solid_von_mises', or 'drag' (alias)."
                )

            # Ensure the driver is treated as a function-like scalar (VecOpInfo on the left)
            # so subsequent arithmetic does not hit unsupported float - VecOpInfo branches
            # in LHS assembly.
            D_mech_n = (_c(0.0) * alpha_n) + D_mech_n

            # Curvature proxy from α^n using only primitive grad/Laplacian operators
            # (keeps compatibility with all backends, including the C++ kernel).
            g_n = grad(alpha_n)
            g2 = inner(g_n, g_n)
            denom = g2 + _c(float(alpha_crack_eta_kappa))
            denom_sqrt = _sqrt(denom)
            # The exact mean curvature κ = div(∇α/|∇α|) is not fully supported in the
            # current symbolic pipeline. We use a robust approximation based on the
            # Laplacian (trace of the Hessian):
            #   κ̃ = (|∇α|^2 Δα) / (|∇α|^2 + η)^{3/2}  ≈ Δα/|∇α|.
            lap_n = Laplacian(alpha_n)
            kappa_n = (g2 * lap_n) / (denom * denom_sqrt)

            D_c = _c(float(alpha_crack_Dc))
            gamma_kappa = _c(float(alpha_crack_gamma_kappa))
            drive = D_mech_n - gamma_kappa * kappa_n - D_c

            # Smooth positive part: <x>_+ ≈ 0.5 (x + sqrt(x^2 + η)).
            eta_pos = _c(float(alpha_crack_eta_pos))
            pos = _c(0.5) * (drive + _sqrt(drive * drive + eta_pos))

            m_pow = float(alpha_crack_m)
            if m_pow < 1.0:
                raise ValueError(f"alpha_crack_m must be >= 1; got {m_pow}.")
            Vc_prev = _c(float(alpha_crack_k)) * (pos ** _c(m_pow))

        # Convert speed [length/time] to a rate [1/time] via (4 ε)^{-1}.
        crack_coef_prev = Vc_prev / (_c(4.0) * eps_alpha_c)

    # Surface-localized erosion/detachment sink is -D_det_prev δ(α) on the RHS,
    # so it enters the residual with a + sign (same convention as the X source).
    surf_coef_prev = D_det_prev + crack_coef_prev

    # Select the advecting velocity for α.
    #
    # NOTE: For the "mix" option we intentionally use *lagged* (C_n,B_n) weights
    # in the alpha transport block. This keeps the current Newton linearization
    # affine in (v, vS, alpha) while still transporting alpha with the same
    # one-domain occupied-volume flux that appears in the incompressibility
    # constraint.
    if support_physics_key == "internal_conversion":
        alpha_advect_default = "biofilm_volume"
    elif support_physics_key == "stored_support":
        alpha_advect_default = "vS"
    else:
        alpha_advect_default = "mix"
    adv_with_key = str(alpha_advect_with or alpha_advect_default).strip().lower()
    if support_physics_key == "internal_conversion" and adv_with_key not in {
        "biofilm",
        "biofilm_volume",
        "biofilm-volume",
        "phase",
        "phase_volume",
        "phase-volume",
    }:
        raise ValueError(
            "support_physics='internal_conversion' requires alpha_advect_with='biofilm_volume' "
            "because alpha tracks the conserved biofilm support."
        )
    if support_physics_key == "stored_support" and adv_with_key not in {
        "vs",
        "v^s",
        "v_s",
        "s",
        "skeleton",
        "solid",
    }:
        raise ValueError(
            "support_physics='stored_support' requires alpha_advect_with='vS' because alpha is a geometric support field."
        )
    if adv_with_key in {"vs", "v^s", "v_s", "s", "skeleton", "solid"}:
        adv_u_k = stored_support_vS_gate_n * vS_k
        adv_u_n = stored_support_vS_gate_n * vS_n
        adv_u_k_comp = [stored_support_vS_gate_n * vS_k[i] for i in range(int(dim))]
        adv_u_n_comp = [stored_support_vS_gate_n * vS_n[i] for i in range(int(dim))]
        dadv_u = None
        dadv_u_comp = [stored_support_vS_gate_n * dvS[i] for i in range(int(dim))]
        div_adv_u_k = stored_support_vS_gate_n * div_vS_k + dot(grad_stored_support_vS_gate_n, vS_k)
        div_adv_u_n = stored_support_vS_gate_n * div_vS_n + dot(grad_stored_support_vS_gate_n, vS_n)
        d_div_adv_u = stored_support_vS_gate_n * div_dvS + dot(grad_stored_support_vS_gate_n, dvS)
    elif adv_with_key in {"v", "fluid"}:
        adv_u_k = v_k
        adv_u_n = v_n
        adv_u_k_comp = [v_k_comp[i] for i in range(int(dim))]
        adv_u_n_comp = [v_n_comp[i] for i in range(int(dim))]
        dadv_u = None
        dadv_u_comp = [dv_comp[i] for i in range(int(dim))]
        div_adv_u_k = div(v_k)
        div_adv_u_n = div(v_n)
        d_div_adv_u = div(dv)
    elif adv_with_key in {"biofilm", "biofilm_volume", "biofilm-volume", "phase", "phase_volume", "phase-volume"}:
        # Total biofilm-volume velocity from the two constituent volume balances:
        #   ∂t(α φ)     + div(α φ v)      = ...
        #   ∂t(α (1-φ)) + div(α (1-φ) vS) = ...
        # Summing gives
        #   ∂t α + div( α [ φ v + (1-φ) vS ] ) = ...
        #
        # Use the current porosity in the k-level flux so the alpha transport
        # Jacobian is fully consistent with the internal-conversion support law.
        # This keeps the conservative alpha block coupled to the current phi/v/vS
        # state instead of relying on a Picard lag.
        if use_B_primary:
            raise ValueError("ratio-free full-state mode does not support alpha_advect_with='biofilm_volume'.")
        one_m_phi_k = _one_minus(phi_k)
        one_m_phi_n = _one_minus(phi_n)
        grad_phi_k = grad(phi_k)
        grad_phi_n = grad(phi_n)
        adv_u_k = phi_k * v_k + one_m_phi_k * vS_k
        adv_u_n = phi_n * v_n + one_m_phi_n * vS_n
        adv_u_k_comp = [phi_k * v_k_comp[i] + one_m_phi_k * vS_k[i] for i in range(int(dim))]
        adv_u_n_comp = [phi_n * v_n_comp[i] + one_m_phi_n * vS_n[i] for i in range(int(dim))]
        dadv_u_comp = [
            phi_k * dv_comp[i] + one_m_phi_k * dvS[i] + dphi * (v_k_comp[i] - vS_k[i])
            for i in range(int(dim))
        ]
        div_adv_u_k = phi_k * div(v_k) + dot(grad_phi_k, v_k) + one_m_phi_k * div_vS_k - dot(grad_phi_k, vS_k)
        div_adv_u_n = phi_n * div(v_n) + dot(grad_phi_n, v_n) + one_m_phi_n * div_vS_n - dot(grad_phi_n, vS_n)
        d_div_adv_u = (
            phi_k * div(dv)
            + dot(grad_phi_k, dv)
            + one_m_phi_k * div_dvS
            - dot(grad_phi_k, dvS)
            + dphi * (div(v_k) - div_vS_k)
            + dot(grad(dphi), v_k - vS_k)
        )
    elif adv_with_key in {"relative", "rel", "v-vs", "v_minus_vs", "slip"}:
        adv_u_k = v_k - vS_k
        adv_u_n = v_n - vS_n
        adv_u_k_comp = [v_k_comp[i] - vS_k[i] for i in range(int(dim))]
        adv_u_n_comp = [v_n_comp[i] - vS_n[i] for i in range(int(dim))]
        dadv_u = None
        dadv_u_comp = [dv_comp[i] - dvS[i] for i in range(int(dim))]
        div_adv_u_k = div(v_k) - div_vS_k
        div_adv_u_n = div(v_n) - div_vS_n
        d_div_adv_u = div(dv) - div_dvS
    elif adv_with_key in {"mix", "mixture", "f", "flux", "volume"}:
        # Legacy mixture/volume transport flux for one-domain alpha:
        #
        #   F = C v + B vS,  with C=(1-alpha)+alpha phi and B=alpha(1-phi).
        #
        # This is the flux constrained by the one-domain incompressibility
        # equation. It is useful for the historical support-creation model, but
        # it is not the physically correct support flux when alpha is supposed
        # to track only the biofilm support.
        adv_u_k = C_n * v_k + B_n * vS_k
        adv_u_n = C_n * v_n + B_n * vS_n
        adv_u_k_comp = [C_n * v_k_comp[i] + B_n * vS_k[i] for i in range(int(dim))]
        adv_u_n_comp = [C_n * v_n_comp[i] + B_n * vS_n[i] for i in range(int(dim))]
        dadv_u = None
        dadv_u_comp = [C_n * dv_comp[i] + B_n * dvS[i] for i in range(int(dim))]

        # Conservative form needs div(F). Expand div(C_n v + B_n vS) explicitly.
        div_adv_u_k = C_n * div(v_k) + dot(gradC_n, v_k) + B_n * div_vS_k + dot(gradB_n, vS_k)
        div_adv_u_n = divCv_n + divBvS_n
        d_div_adv_u = C_n * div(dv) + dot(gradC_n, dv) + B_n * div_dvS + dot(gradB_n, dvS)
    elif adv_with_key in {"mid", "avg", "average", "mean", "interface", "midpoint", "half_sum"}:
        adv_u_k = _c(0.5) * (v_k + vS_k)
        adv_u_n = _c(0.5) * (v_n + vS_n)
        adv_u_k_comp = [_c(0.5) * (v_k_comp[i] + vS_k[i]) for i in range(int(dim))]
        adv_u_n_comp = [_c(0.5) * (v_n_comp[i] + vS_n[i]) for i in range(int(dim))]
        dadv_u = None
        dadv_u_comp = [_c(0.5) * (dv_comp[i] + dvS[i]) for i in range(int(dim))]
        div_adv_u_k = _c(0.5) * (div(v_k) + div_vS_k)
        div_adv_u_n = _c(0.5) * (div(v_n) + div_vS_n)
        d_div_adv_u = _c(0.5) * (div(dv) + div_dvS)
    elif adv_with_key in {"mix_biofilm", "mix-biofilm", "mix_bio", "mixbio", "mix_cutoff", "mixcutoff"}:
        # Mixture/volume velocity but *gated* so we do not advect α through the pure fluid
        # where α≈0 (which can create a spurious thin "chimney" when v has strong outflow).
        #
        # We gate only the C-part by a smooth cutoff g(α^n), leaving the biofilm-side
        # velocity unchanged for α above the cutoff:
        #   F_adv = (g C) v + B vS,   g(α) = α^m / (α^m + α0^m).
        alpha0 = float(alpha_mix_gate_alpha0)
        if not (0.0 < alpha0 < 1.0):
            raise ValueError(f"alpha_mix_gate_alpha0 must be in (0,1); got {alpha0}.")
        m_pow = int(alpha_mix_gate_power)
        if m_pow < 1:
            raise ValueError(f"alpha_mix_gate_power must be >= 1; got {m_pow}.")

        # Build α^m with repeated multiplication (keeps backend compatibility).
        a_pow = alpha_n
        for _ in range(m_pow - 1):
            a_pow = a_pow * alpha_n
        a0_pow = _c(alpha0 ** float(m_pow))
        gate_n = a_pow / (a_pow + a0_pow + _c(1.0e-12))

        Cg_n = gate_n * C_n
        gradCg_n = grad(Cg_n)

        adv_u_k = Cg_n * v_k + B_n * vS_k
        adv_u_n = Cg_n * v_n + B_n * vS_n
        adv_u_k_comp = [Cg_n * v_k_comp[i] + B_n * vS_k[i] for i in range(int(dim))]
        adv_u_n_comp = [Cg_n * v_n_comp[i] + B_n * vS_n[i] for i in range(int(dim))]
        dadv_u = None
        dadv_u_comp = [Cg_n * dv_comp[i] + B_n * dvS[i] for i in range(int(dim))]

        div_adv_u_k = Cg_n * div(v_k) + dot(gradCg_n, v_k) + B_n * div_vS_k + dot(gradB_n, vS_k)
        div_adv_u_n = Cg_n * div(v_n) + dot(gradCg_n, v_n) + divBvS_n
        d_div_adv_u = Cg_n * div(dv) + dot(gradCg_n, dv) + B_n * div_dvS + dot(gradB_n, dvS)
    else:
        raise ValueError(
            f"Unknown alpha_advect_with={alpha_advect_with!r}. "
            "Use 'vS' (default), 'v', 'biofilm_volume', 'relative', 'mix', 'interface', or 'mix_biofilm'."
        )

    adv_key = str(alpha_advection_form).strip().lower()
    if adv_key in {"advective", "nonconservative", "v.grad", "v·grad", "vgrad"}:
        adv_key = "advective"
    elif adv_key in {"conservative", "div", "divergence", "div(alpha*v)"}:
        adv_key = "conservative"
    elif adv_key in {
        "conservative_weak",
        "conservative-weak",
        "conservative_ibp",
        "conservative-ibp",
        "ibp",
        "weak",
        "weak_conservative",
        "weak-conservative",
    }:
        adv_key = "conservative_weak"
    elif adv_key in {
        "interface_band_conservative",
        "interface-band-conservative",
        "band_conservative",
        "band-conservative",
        "interface_conservative",
        "interface-conservative",
        "localized_interface",
        "localized-interface",
    }:
        adv_key = "interface_band_conservative"
    else:
        raise ValueError(
            f"Unknown alpha_advection_form={alpha_advection_form!r}. Use 'advective', 'conservative', 'conservative_weak', or 'interface_band_conservative'."
        )

    band_alpha_k = _c(4.0) * alpha_k * _one_minus(alpha_k)
    band_alpha_n = _c(4.0) * alpha_n * _one_minus(alpha_n)
    dband_alpha = _c(4.0) * dalpha * _one_minus(_c(2.0) * alpha_k)
    grad_band_alpha_k = _c(4.0) * _one_minus(_c(2.0) * alpha_k) * grad(alpha_k)
    grad_band_alpha_n = _c(4.0) * _one_minus(_c(2.0) * alpha_n) * grad(alpha_n)
    d_grad_band_alpha = _c(4.0) * (
        _one_minus(_c(2.0) * alpha_k) * grad(dalpha)
        - _c(2.0) * dalpha * grad(alpha_k)
    )

    if support_physics_key == "internal_conversion" and adv_key not in {"conservative", "conservative_weak"}:
        raise ValueError(
            "support_physics='internal_conversion' requires a conservative alpha law. "
            "Use alpha_advection_form='conservative_weak' (recommended) or 'conservative'."
        )
    if support_physics_key == "stored_support" and adv_key != "advective":
        raise ValueError(
            "support_physics='stored_support' requires alpha_advection_form='advective' because alpha follows interface kinematics."
        )

    if adv_key == "advective":
        # Indicator advection: ∂t α + u·∇α = ...
        adv_alpha_k = dot(grad(alpha_k), adv_u_k)
        adv_alpha_n = dot(grad(alpha_n), adv_u_n)
    elif adv_key == "conservative":
        # Conservative advection: ∂t α + div(α u) = ...
        #
        # We implement div(α u) as u·∇α + α div(u) to avoid relying on backend
        # support for `div(alpha*u)` in all code paths.
        adv_alpha_k = dot(grad(alpha_k), adv_u_k) + alpha_k * div_adv_u_k
        adv_alpha_n = dot(grad(alpha_n), adv_u_n) + alpha_n * div_adv_u_n
    elif adv_key == "interface_band_conservative":
        # Localized interface transport:
        #   ∂t b(α) + div(b(α) u_I) = ...
        # with b(α)=4α(1-α), u_I = 0.5 (v + vS).
        adv_alpha_k = dot(grad_band_alpha_k, adv_u_k) + band_alpha_k * div_adv_u_k
        adv_alpha_n = dot(grad_band_alpha_n, adv_u_n) + band_alpha_n * div_adv_u_n
    elif adv_key == "conservative_weak":
        # SUPG/CIP stabilization uses the strong conservative residual even
        # when the Galerkin form is integrated by parts.
        adv_alpha_k = dot(grad(alpha_k), adv_u_k) + alpha_k * div_adv_u_k
        adv_alpha_n = dot(grad(alpha_n), adv_u_n) + alpha_n * div_adv_u_n
    else:
        adv_alpha_k = None
        adv_alpha_n = None

    time_alpha_k = alpha_k
    time_alpha_n = alpha_n
    if adv_key == "interface_band_conservative":
        time_alpha_k = band_alpha_k
        time_alpha_n = band_alpha_n

    alpha_source_k = _c(0.0)
    alpha_source_n = _c(0.0)
    if support_physics_key == "legacy_exchange":
        alpha_source_k = -G_k * alpha_k * _one_minus(alpha_k) + surf_coef_prev * delta_k
        alpha_source_n = -G_n * alpha_n * _one_minus(alpha_n) + surf_coef_prev * delta_n

    f_alpha_k = (time_alpha_k - time_alpha_n) * inv_dt
    if adv_key == "conservative_weak":
        r_alpha = alpha_test * f_alpha_k * dx
        # Weak conservative transport:
        #   (∂t α, w) - (α u, ∇w) = 0
        #
        # The omitted boundary term represents the physical alpha flux through
        # the outer boundary. It vanishes for natural no-flux boundaries, and it
        # also vanishes on Dirichlet segments because the admissible test
        # functions are zero there.
        flux_dot_grad_test_k = _c(0.0)
        flux_dot_grad_test_n = _c(0.0)
        for i in range(int(dim)):
            flux_dot_grad_test_k = flux_dot_grad_test_k + (alpha_k * adv_u_k_comp[i]) * grad(alpha_test)[i]
            flux_dot_grad_test_n = flux_dot_grad_test_n + (alpha_n * adv_u_n_comp[i]) * grad(alpha_test)[i]
        r_alpha += -th * flux_dot_grad_test_k * dx
        r_alpha += -one_m_th * flux_dot_grad_test_n * dx
        if ds_alpha_transport is not None:
            n_b = FacetNormal()
            flux_alpha_bdry_k = _c(0.0)
            flux_alpha_bdry_n = _c(0.0)
            for i in range(int(dim)):
                n_i = _vector_component(n_b, i)
                flux_alpha_bdry_k += (alpha_k * adv_u_k_comp[i]) * n_i
                flux_alpha_bdry_n += (alpha_n * adv_u_n_comp[i]) * n_i
            r_alpha += th * alpha_test * flux_alpha_bdry_k * ds_alpha_transport
            r_alpha += one_m_th * alpha_test * flux_alpha_bdry_n * ds_alpha_transport
        r_alpha += alpha_test * th * alpha_source_k * dx
        r_alpha += alpha_test * one_m_th * alpha_source_n * dx
    else:
        f_alpha_k += th * (adv_alpha_k + alpha_source_k)
        f_alpha_k += one_m_th * (adv_alpha_n + alpha_source_n)
        r_alpha = alpha_test * f_alpha_k * dx
    r_alpha += D_alpha_c * inner(grad(alpha_k), grad(alpha_test)) * dx

    # Conservative interface-maintenance flux for α.
    #
    # This is a geometry regularizer, not a physical mixing term:
    #   F_reg = γ [ α(1-α) n
    #               - ε_n ( (∇α·n) n )
    #               - ε_t ( ∇α - (∇α·n) n ) ]
    # with lagged interface normal n = ∇α^n / |∇α^n|.
    #
    # Because it is written in divergence form and enters weakly as
    #   -∫ F_reg · ∇w dx,
    # it has zero contribution to d/dt ∫ α dx at the continuous level under
    # the natural no-flux boundary condition.
    alpha_reg_key = str(alpha_interface_reg).strip().lower()
    if alpha_reg_key in {"none", "off", "disabled", "0"}:
        alpha_reg_key = "none"
    elif alpha_reg_key in {"olsson_nt", "olsson-nt", "olsson_nt_projected", "normal_tangent", "normal+tangent"}:
        alpha_reg_key = "olsson_nt"
    else:
        raise ValueError(
            f"Unknown alpha_interface_reg={alpha_interface_reg!r}. Use 'none' or 'olsson_nt'."
        )

    alpha_reg_gamma_c = _c(float(alpha_interface_reg_gamma))
    if alpha_reg_key == "olsson_nt" and float(alpha_interface_reg_gamma) != 0.0:
        eta_n = float(alpha_interface_reg_eta)
        if eta_n <= 0.0:
            raise ValueError(f"alpha_interface_reg_eta must be > 0; got {eta_n}.")
        eps_n_c = _c(float(alpha_interface_reg_eps_normal))
        eps_t_c = _c(float(alpha_interface_reg_eps_tangent))

        g_alpha_n = grad(alpha_n)
        g_alpha_k = grad(alpha_k)
        g_alpha_test = grad(alpha_test)
        g_alpha_d = grad(dalpha)

        n_norm = _sqrt(inner(g_alpha_n, g_alpha_n) + _c(eta_n))
        n_comp = [g_alpha_n[i] / n_norm for i in range(int(dim))]

        gk_dot_n = _c(0.0)
        gd_dot_n = _c(0.0)
        gt_dot_n = _c(0.0)
        for i in range(int(dim)):
            gk_dot_n = gk_dot_n + g_alpha_k[i] * n_comp[i]
            gd_dot_n = gd_dot_n + g_alpha_d[i] * n_comp[i]
            gt_dot_n = gt_dot_n + g_alpha_test[i] * n_comp[i]

        # Weak divergence form: -∫ F_reg · ∇w dx
        r_alpha += -alpha_reg_gamma_c * alpha_k * _one_minus(alpha_k) * gt_dot_n * dx
        r_alpha += alpha_reg_gamma_c * eps_n_c * gk_dot_n * gt_dot_n * dx
        r_alpha += alpha_reg_gamma_c * eps_t_c * (inner(g_alpha_k, g_alpha_test) - gk_dot_n * gt_dot_n) * dx

    # Cahn–Hilliard regularization: α_t + ... = div( M(α) ∇μ_α ), μ_α = γ(-εΔα + (1/ε)W'(α)).
    ch_enabled = float(alpha_ch_M) != 0.0 and float(alpha_ch_gamma) != 0.0
    if bool(alpha_cahn_conservative) and ch_enabled:
        raise ValueError("alpha_cahn_conservative cannot be used together with Cahn–Hilliard regularization (alpha_ch_*).")

    # Allen–Cahn regularization: -(Mγ ε Δα) + (Mγ/ε) W'(α) in the residual.
    ac_enabled = float(alpha_cahn_M) != 0.0 and float(alpha_cahn_gamma) != 0.0
    if ac_enabled and ch_enabled:
        raise ValueError("Allen–Cahn (alpha_cahn_*) and Cahn–Hilliard (alpha_ch_*) cannot both be enabled simultaneously.")
    if support_physics_key == "internal_conversion" and ac_enabled and not bool(alpha_cahn_conservative):
        raise ValueError(
            "support_physics='internal_conversion' preserves total alpha, so nonconservative Allen–Cahn regularization is incompatible. "
            "Use Cahn–Hilliard, Olsson-type conservative regularization, or conservative Allen–Cahn."
        )
    if bool(alpha_cahn_conservative):
        cons_mode = str(alpha_cahn_conservative_mode).strip().lower()
        if cons_mode in {"solve", "unknown", "lagrange", "constraint"}:
            cons_mode = "unknown"
        elif cons_mode in {"eliminate", "elim", "project", "projected", "explicit"}:
            cons_mode = "eliminate"
        else:
            raise ValueError(
                f"Unknown alpha_cahn_conservative_mode {alpha_cahn_conservative_mode!r}. Use 'unknown' or 'eliminate'."
            )
        if not ac_enabled:
            raise ValueError("alpha_cahn_conservative=True requires alpha_cahn_M and alpha_cahn_gamma to be nonzero.")
    else:
        cons_mode = "unknown"
    if ac_enabled:
        # W'(α) for W(α)=α^2(1-α)^2 is 2α(1-α)(1-2α).
        Wp_k = _c(2.0) * alpha_k * _one_minus(alpha_k) * _one_minus(_c(2.0) * alpha_k)
        mob_key = str(alpha_cahn_mobility).strip().lower()
        if mob_key in {"constant", "const"}:
            M_ac_k = M_alpha_c
            dM_ac = _c(0.0) * dalpha
            mob_prime_k = _c(0.0) * alpha_k
        elif mob_key in {"degenerate", "deg", "alpha(1-alpha)", "alpha*(1-alpha)", "a(1-a)"}:
            # Degenerate mobility: M(α)=M0 α(1-α)
            floor = float(alpha_cahn_mobility_floor)
            if floor < 0.0:
                raise ValueError(f"alpha_cahn_mobility_floor must be >= 0; got {floor}.")
            mob_k = alpha_k * _one_minus(alpha_k) + _c(floor)
            mob_prime_k = _one_minus(_c(2.0) * alpha_k)  # d/dα [α(1-α)] = 1-2α
            M_ac_k = M_alpha_c * mob_k
            dM_ac = M_alpha_c * mob_prime_k * dalpha
        else:
            raise ValueError(f"Unknown alpha_cahn_mobility {alpha_cahn_mobility!r}. Use 'constant' or 'degenerate'.")

        M_gamma_k = gamma_alpha_c * M_ac_k
        dM_gamma = gamma_alpha_c * dM_ac

        # Strong term: +M(α) μ_α, μ_α=γ(-εΔα + (1/ε)W'(α)).
        # Integrate by parts the Laplacian term; if M depends on α, an extra
        # interface-localized term ε ∇M·∇α appears.
        r_alpha += (M_gamma_k * eps_alpha_c) * inner(grad(alpha_k), grad(alpha_test)) * dx
        if mob_key not in {"constant", "const"}:
            r_alpha += alpha_test * ((eps_alpha_c * gamma_alpha_c * M_alpha_c) * mob_prime_k * inner(grad(alpha_k), grad(alpha_k))) * dx
        r_alpha += alpha_test * ((M_gamma_k / eps_alpha_c) * Wp_k) * dx

        if bool(alpha_cahn_conservative):
            if lambda_alpha_k is None:
                raise ValueError("alpha_cahn_conservative=True requires lambda_alpha_k to be provided.")
            r_alpha += -alpha_test * (M_ac_k * lambda_alpha_k) * dx

    r_mu_alpha = None
    a_mu_alpha = None
    if ch_enabled:
        if mu_alpha_k is None or mu_alpha_n is None or dmu_alpha is None or mu_alpha_test is None:
            raise ValueError(
                "Cahn–Hilliard regularization requires (mu_alpha_k, mu_alpha_n, dmu_alpha, mu_alpha_test) to be provided."
            )
        eps_ch_val = float(alpha_ch_eps)
        if eps_ch_val <= 0.0:
            raise ValueError(f"alpha_ch_eps must be > 0 when Cahn–Hilliard is enabled; got {eps_ch_val}.")
        eps_ch_c = _as_constant(alpha_ch_eps)
        M_ch_c = _as_constant(alpha_ch_M)
        gamma_ch_c = _as_constant(alpha_ch_gamma)

        # Double-well derivative W'(α) for W(α)=α^2(1-α)^2 is 2α(1-α)(1-2α).
        Wp_ch_k = _c(2.0) * alpha_k * _one_minus(alpha_k) * _one_minus(_c(2.0) * alpha_k)
        # W''(α) = 2 - 12α + 12α^2
        Wpp_ch_k = (-_c(12.0) * alpha_k) + (_c(12.0) * (alpha_k * alpha_k)) + _c(2.0)

        mob_key = str(alpha_ch_mobility).strip().lower()
        if mob_key in {"constant", "const"}:
            M_ch_k = M_ch_c
            M_ch_n = M_ch_c
            dM_ch = _c(0.0) * dalpha
        elif mob_key in {"degenerate", "deg", "alpha(1-alpha)", "alpha*(1-alpha)", "a(1-a)"}:
            # Degenerate mobility: M(α)=M0 α(1-α)
            mob_k = alpha_k * _one_minus(alpha_k)
            mob_n = alpha_n * _one_minus(alpha_n)
            mob_prime_k = _one_minus(_c(2.0) * alpha_k)  # d/dα [α(1-α)] = 1-2α
            M_ch_k = M_ch_c * mob_k
            M_ch_n = M_ch_c * mob_n
            dM_ch = M_ch_c * mob_prime_k * dalpha
        else:
            raise ValueError(f"Unknown alpha_ch_mobility {alpha_ch_mobility!r}. Use 'constant' or 'degenerate'.")

        # α equation: +∫ M(α) ∇μ · ∇w  (no-flux boundary)
        #
        # IMPORTANT: keep GradOpInfo on the left in scalar×grad products for backend
        # compatibility (function×grad(function) is not implemented, but grad(function)×function is).
        r_alpha += th * inner(grad(mu_alpha_k) * M_ch_k, grad(alpha_test)) * dx
        r_alpha += one_m_th * inner(grad(mu_alpha_n) * M_ch_n, grad(alpha_test)) * dx

        # μ equation: ∫ ψ ( μ - γ(-εΔα + (1/ε)W'(α)) ) dx = 0 (drop boundary term).
        r_mu_alpha = mu_alpha_test * mu_alpha_k * dx
        r_mu_alpha += -(gamma_ch_c * eps_ch_c) * inner(grad(alpha_k), grad(mu_alpha_test)) * dx
        r_mu_alpha += -mu_alpha_test * ((gamma_ch_c / eps_ch_c) * Wp_ch_k) * dx
    r_alpha += -alpha_test * f_alpha * dx

    # Jacobian (k-part only)
    # δG
    d_alpha_source_k = _c(0.0)
    if support_physics_key == "legacy_exchange":
        dG = (_c(float(k_g)) * _one_minus(phi_k) * (dmon_dS * dS) + (-_c(float(k_g)) * mon_k) * dphi)
        # δ[ G α(1-α) ] = (δG) α(1-α) + G (1-2α) δα
        dalpha_logistic = _one_minus(_c(2.0) * alpha_k) * dalpha
        # δ[ (D_det_prev + crack_coef_prev) * δ(α) ] = (D_det_prev + crack_coef_prev) * δ'(α) δα
        # (surf coefficients are lagged).
        d_delta_k = _c(4.0) * (_one_minus(_c(2.0) * alpha_k)) * dalpha
        d_surf = surf_coef_prev * d_delta_k
        d_alpha_source_k = -(dG * alpha_k * _one_minus(alpha_k) + G_k * dalpha_logistic) + d_surf

    grad_alpha_k_dot_dadv_u = _c(0.0)
    grad_band_alpha_k_dot_dadv_u = _c(0.0)
    for i in range(int(dim)):
        grad_alpha_k_dot_dadv_u = grad_alpha_k_dot_dadv_u + grad(alpha_k)[i] * dadv_u_comp[i]
        grad_band_alpha_k_dot_dadv_u = grad_band_alpha_k_dot_dadv_u + grad_band_alpha_k[i] * dadv_u_comp[i]

    if adv_key == "interface_band_conservative":
        a_alpha = alpha_test * (dband_alpha * inv_dt) * dx
    else:
        a_alpha = alpha_test * (dalpha * inv_dt) * dx
    if adv_key == "advective":
        a_alpha += alpha_test * th * (grad_alpha_k_dot_dadv_u + dot(grad(dalpha), adv_u_k)) * dx
    elif adv_key == "conservative":
        a_alpha += alpha_test * th * (
            grad_alpha_k_dot_dadv_u
            + dot(grad(dalpha), adv_u_k)
            + dalpha * div_adv_u_k
            + alpha_k * d_div_adv_u
        ) * dx
    elif adv_key == "interface_band_conservative":
        a_alpha += alpha_test * th * (
            grad_band_alpha_k_dot_dadv_u
            + dot(d_grad_band_alpha, adv_u_k)
            + dband_alpha * div_adv_u_k
            + band_alpha_k * d_div_adv_u
        ) * dx
    else:
        dflux_dot_grad_test_k = _c(0.0)
        for i in range(int(dim)):
            dflux_i = dalpha * adv_u_k_comp[i] + alpha_k * dadv_u_comp[i]
            dflux_dot_grad_test_k = dflux_dot_grad_test_k + dflux_i * grad(alpha_test)[i]
        a_alpha += -th * dflux_dot_grad_test_k * dx
        if ds_alpha_transport is not None:
            n_b = FacetNormal()
            dflux_alpha_bdry_k = _c(0.0)
            for i in range(int(dim)):
                n_i = _vector_component(n_b, i)
                dflux_alpha_bdry_k += (dalpha * adv_u_k_comp[i] + alpha_k * dadv_u_comp[i]) * n_i
            a_alpha += th * alpha_test * dflux_alpha_bdry_k * ds_alpha_transport
    a_alpha += alpha_test * th * d_alpha_source_k * dx
    a_alpha += D_alpha_c * inner(grad(dalpha), grad(alpha_test)) * dx
    if alpha_reg_key == "olsson_nt" and float(alpha_interface_reg_gamma) != 0.0:
        a_alpha += -alpha_reg_gamma_c * _one_minus(_c(2.0) * alpha_k) * dalpha * gt_dot_n * dx
        a_alpha += alpha_reg_gamma_c * eps_n_c * gd_dot_n * gt_dot_n * dx
        a_alpha += alpha_reg_gamma_c * eps_t_c * (inner(g_alpha_d, g_alpha_test) - gd_dot_n * gt_dot_n) * dx
    if ac_enabled:
        # W''(α) = 2 - 12α + 12α^2
        # NOTE: keep the function-like term on the left; float - VecOpInfo is not supported.
        Wpp_k = (-_c(12.0) * alpha_k) + (_c(12.0) * (alpha_k * alpha_k)) + _c(2.0)
        if mob_key in {"constant", "const"}:
            a_alpha += (M_gamma_alpha * eps_alpha_c) * inner(grad(dalpha), grad(alpha_test)) * dx
            a_alpha += alpha_test * ((M_gamma_alpha / eps_alpha_c) * Wpp_k * dalpha) * dx
            if bool(alpha_cahn_conservative) and cons_mode == "unknown":
                if dlambda_alpha is None:
                    raise ValueError(
                        "alpha_cahn_conservative_mode='unknown' requires dlambda_alpha (lambda_alpha as unknown)."
                    )
                a_alpha += -alpha_test * (M_alpha_c * dlambda_alpha) * dx
        else:
            # d[ ε Mγ ∇α·∇w ] = ε (dMγ ∇α·∇w + Mγ ∇δα·∇w)
            a_alpha += (eps_alpha_c * dM_gamma) * inner(grad(alpha_k), grad(alpha_test)) * dx
            a_alpha += (M_gamma_k * eps_alpha_c) * inner(grad(dalpha), grad(alpha_test)) * dx

            # d[ ε w (∇Mγ·∇α) ] with M= M0 α(1-α): ∇Mγ = γ M0 (1-2α) ∇α
            g2 = inner(grad(alpha_k), grad(alpha_k))
            a_alpha += alpha_test * ((eps_alpha_c * gamma_alpha_c * M_alpha_c) * ((-_c(2.0) * dalpha) * g2)) * dx
            a_alpha += alpha_test * ((eps_alpha_c * gamma_alpha_c * M_alpha_c) * (mob_prime_k * (_c(2.0) * inner(grad(alpha_k), grad(dalpha))))) * dx

            # d[ w (Mγ/ε) W'(α) ] = w/ε (dMγ W'(α) + Mγ W''(α) δα)
            a_alpha += alpha_test * (((dM_gamma / eps_alpha_c) * Wp_k) + ((M_gamma_k / eps_alpha_c) * Wpp_k * dalpha)) * dx
            if bool(alpha_cahn_conservative):
                if lambda_alpha_k is None:
                    raise ValueError("alpha_cahn_conservative=True requires lambda_alpha_k to be provided.")
                a_alpha += -alpha_test * (dM_ac * lambda_alpha_k) * dx
                if cons_mode == "unknown":
                    if dlambda_alpha is None:
                        raise ValueError(
                            "alpha_cahn_conservative_mode='unknown' requires dlambda_alpha (lambda_alpha as unknown)."
                        )
                    a_alpha += -alpha_test * (M_ac_k * dlambda_alpha) * dx

    if ch_enabled:
        # k-part Jacobian of ∫ M(α) ∇μ · ∇w
        a_alpha += th * (dM_ch * inner(grad(mu_alpha_k), grad(alpha_test))) * dx
        a_alpha += th * inner(M_ch_k * grad(dmu_alpha), grad(alpha_test)) * dx

        # Jacobian of μ equation.
        a_mu_alpha = mu_alpha_test * dmu_alpha * dx
        a_mu_alpha += -(gamma_ch_c * eps_ch_c) * inner(grad(dalpha), grad(mu_alpha_test)) * dx
        a_mu_alpha += -mu_alpha_test * ((gamma_ch_c / eps_ch_c) * Wpp_ch_k * dalpha) * dx
    elif mu_alpha_k is not None and dmu_alpha is not None and mu_alpha_test is not None and float(alpha_mu_aux_pin) != 0.0:
        mu_aux_pin_c = _c(float(alpha_mu_aux_pin))
        r_mu_alpha = mu_aux_pin_c * mu_alpha_test * mu_alpha_k * dx
        a_mu_alpha = mu_aux_pin_c * mu_alpha_test * dmu_alpha * dx

    # Optional consistent stabilization for advection-dominated α (useful with D_alpha=0).
    # - SUPG: τ (u·∇w) R_alpha
    # - CIP:  γ h^3 (1/dt + |u|/h) <[∂_n α],[∂_n w]>_F on interior facets
    #
    # Notes:
    # - We use a lagged advector u_n in τ and in the test-direction to keep the Jacobian
    #   consistent (no extra u-coupling from the stabilization weights).
    # - CIP only affects regions where ∇α is non-zero (i.e. the diffuse interface),
    #   and remains consistent because [∂_n α]=0 for smooth α.
    if float(alpha_supg) != 0.0:
        h_a = MeshSize()
        vmag2 = _c(0.0)
        for j in range(int(dim)):
            vmag2 += adv_u_n_comp[j] * adv_u_n_comp[j]
        vmag = _sqrt(vmag2 + _c(1.0e-12))
        denom = (_c(2.0) * inv_dt) * (_c(2.0) * inv_dt) + (_c(2.0) * vmag / (h_a + _c(1.0e-12))) * (
            _c(2.0) * vmag / (h_a + _c(1.0e-12))
        )
        tau_supg = _c(float(alpha_supg)) / _sqrt(denom + _c(1.0e-16))
        # Avoid `dot(grad(test), v)` due to current dot/left_dot limitations in the
        # C++ backend for some mixed expression types; expand componentwise.
        g_test = grad(alpha_test)
        w_supg = _c(0.0)
        for j in range(int(dim)):
            w_supg += g_test[j] * adv_u_n_comp[j]
        f_alpha_supg_k = (time_alpha_k - time_alpha_n) * inv_dt
        f_alpha_supg_k += th * (adv_alpha_k + alpha_source_k)
        f_alpha_supg_k += one_m_th * (adv_alpha_n + alpha_source_n)
        r_alpha += tau_supg * w_supg * f_alpha_supg_k * dx

        df_alpha_k = dalpha * inv_dt
        if adv_key == "interface_band_conservative":
            df_alpha_k = dband_alpha * inv_dt
        if adv_key == "advective":
            df_alpha_k += th * (
                grad_alpha_k_dot_dadv_u
                + dot(grad(dalpha), adv_u_k)
                + d_alpha_source_k
            )
        elif adv_key == "interface_band_conservative":
            df_alpha_k += th * (
                grad_band_alpha_k_dot_dadv_u
                + dot(d_grad_band_alpha, adv_u_k)
                + dband_alpha * div_adv_u_k
                + band_alpha_k * d_div_adv_u
                + d_alpha_source_k
            )
        else:
            df_alpha_k += th * (
                grad_alpha_k_dot_dadv_u
                + dot(grad(dalpha), adv_u_k)
                + dalpha * div_adv_u_k
                + alpha_k * d_div_adv_u
                + d_alpha_source_k
            )
        a_alpha += tau_supg * w_supg * df_alpha_k * dx

    if float(alpha_cip) != 0.0 and ds_cip is not None:
        n_int = FacetNormal()
        h_F = avg(MeshSize())
        scale = inv_dt
        tau_cip = _c(float(alpha_cip)) * (h_F * h_F * h_F) * scale
        r_alpha += tau_cip * _grad_inner_jump(alpha_k, alpha_test, n_int) * ds_cip
        a_alpha += tau_cip * _grad_inner_jump(dalpha, alpha_test, n_int) * ds_cip

    # ------------------------------------------------------------------
    # (v-a) Conservative Allen–Cahn constraint (optional): determine λ_α(t)
    # ------------------------------------------------------------------
    r_alpha_lambda = None
    a_alpha_lambda = None
    if bool(alpha_cahn_conservative) and cons_mode == "unknown":
        if not ac_enabled:
            raise ValueError("alpha_cahn_conservative=True requires alpha_cahn_M and alpha_cahn_gamma to be nonzero.")
        if lambda_alpha_k is None or dlambda_alpha is None or lambda_alpha_test is None:
            raise ValueError(
                "alpha_cahn_conservative=True requires (lambda_alpha_k, dlambda_alpha, lambda_alpha_test) to be provided."
            )

        # Constraint: ∫ M(α) (μ_α - λ_α) dx = 0, with μ_α = γ(-ε Δα + (1/ε)W'(α)).
        #
        # IMPORTANT: avoid assembling Laplacian(alpha) directly in the constraint.
        # For the domain integral of M(-εΔα), use integration by parts and drop the
        # boundary term (consistent with the natural no-flux condition for α used
        # in the biofilm model):
        #   ∫ M(-εΔα) dx = ε ∫ ∇M·∇α dx.
        #
        # This uses only first derivatives and is numerically more robust for high-order runs.
        lam_scale = alpha_cahn_lambda_scale if alpha_cahn_lambda_scale is not None else _c(1.0)
        r_alpha_lambda = lam_scale * lambda_alpha_test * (
            ((gamma_alpha_c / eps_alpha_c) * (M_ac_k * Wp_k)) - (M_ac_k * lambda_alpha_k)
        ) * dx

        # Domain correction: γ ε ∇M·∇α (only nonzero for variable mobility).
        if mob_key not in {"constant", "const"}:
            g2 = inner(grad(alpha_k), grad(alpha_k))
            r_alpha_lambda += lam_scale * lambda_alpha_test * (
                (eps_alpha_c * gamma_alpha_c * M_alpha_c) * mob_prime_k * g2
            ) * dx

        # Jacobian (k-part only)
        # d[ γ(M/ε)W'(α) - M λ ]
        a_alpha_lambda = lam_scale * lambda_alpha_test * (
            (gamma_alpha_c / eps_alpha_c) * ((dM_ac * Wp_k) + (M_ac_k * Wpp_k * dalpha)) + (-(dM_ac * lambda_alpha_k) - (M_ac_k * dlambda_alpha))
        ) * dx

        if mob_key not in {"constant", "const"}:
            g2 = inner(grad(alpha_k), grad(alpha_k))
            a_alpha_lambda += lam_scale * lambda_alpha_test * (
                (eps_alpha_c * gamma_alpha_c * M_alpha_c) * ((-_c(2.0) * dalpha) * g2 + mob_prime_k * (_c(2.0) * inner(grad(alpha_k), grad(dalpha))))
            ) * dx

    # ------------------------------------------------------------------
    # (v-b) Bulk damage evolution (optional): cohesion loss driven by solid stress
    # ------------------------------------------------------------------
    r_damage = None
    a_damage = None
    if use_damage:
        damage_model_key = str(damage_model).strip().lower()
        D_d_c = _c(float(damage_D))
        gamma_out_c = _c(float(damage_gamma_out))
        # Lagged von Mises driver from previous skeleton state u_n.
        # Used by both damage models below; lagging keeps Newton tangents robust.
        sigma_vm = _c(0.0)
        drive_vm = _c(0.0)
        if solid_model_key in {"linear", "small_strain", "linear_elastic"}:
            eps_un = _epsilon(u_n)
            sig_un = _c(2.0) * mu_s * eps_un + lambda_s * div(u_n) * Identity(int(dim))
        elif solid_model_key in {"hencky", "hencky_log", "hencky_log_strain"}:
            sig_un = sigma_hencky(u_n, mu_s, lambda_s, dim=int(dim))
        elif solid_model_key in {"stvk", "svk", "saint_venant_kirchhoff", "saint-venant-kirchhoff"}:
            sig_un = sigma_svk(u_n, mu_s, lambda_s, dim=int(dim))
        elif _is_seboldt_neo_hookean_model(solid_model_key):
            sig_un = sigma_neo_hookean_seboldt(u_n, mu_s, lambda_s, dim=int(dim))
        else:
            if c_nh is None:
                c_nh = mu_s / _c(2.0)
            if beta_nh is None:
                beta_nh = lambda_s / (_c(2.0) * mu_s)
            sig_un = sigma_neo_hookean(u_n, c_nh, beta_nh, dim=int(dim))

        tr_sig = trace(sig_un)
        s_dev = sig_un - (tr_sig / _c(float(dim))) * Identity(int(dim))
        vm2 = _c(1.5) * inner(s_dev, s_dev)
        sigma_vm = _sqrt(vm2 + _c(1.0e-16))

        if float(damage_sigma_cr) > 0.0:
            sigma_cr_c = _c(float(damage_sigma_cr))
            ratio = sigma_vm / sigma_cr_c - _c(1.0)
            pos_ratio = _smooth_pos(ratio, eta=float(damage_eta_pos))
            drive_vm = pos_ratio ** _c(float(damage_m))
        else:
            drive_vm = sigma_vm

        if damage_model_key in {"kinetic", "legacy"}:
            # Legacy advection-reaction-diffusion model:
            #   α (∂t d + vS·∇d) - α rate (1-d) - div(D_d ∇d) + γ_out (1-α)^16 d = 0.
            rate = _c(0.0)
            if float(damage_k) != 0.0:
                rate = _c(float(damage_k)) * drive_vm

            f_dmg_k = alpha_k * ((d_k - d_n) * inv_dt)
            f_dmg_k += th * alpha_k * dot(grad(d_k), vS_k) + one_m_th * alpha_n * dot(grad(d_n), vS_n)
            f_dmg_k += -alpha_k * rate * one_m_d_k

            r_damage = d_test * f_dmg_k * dx
            r_damage += D_d_c * inner(grad(d_k), grad(d_test)) * dx
            if float(damage_gamma_out) != 0.0:
                r_damage += gamma_out_c * w_phi_fluid_k * d_k * d_test * dx

            # Jacobian (k-part only)
            df_dmg_k = dalpha * ((d_k - d_n) * inv_dt) + alpha_k * (dd * inv_dt)
            df_dmg_k += th * (dalpha * dot(grad(d_k), vS_k) + alpha_k * (dot(grad(dd), vS_k) + dot(grad(d_k), dvS)))
            df_dmg_k += -dalpha * rate * one_m_d_k + alpha_k * rate * dd

            a_damage = d_test * df_dmg_k * dx
            a_damage += D_d_c * inner(grad(dd), grad(d_test)) * dx
            if float(damage_gamma_out) != 0.0:
                # δ[ w(α) d ] = w δd + (δw) d, with δw = dw_phi_fluid_k.
                a_damage += gamma_out_c * (w_phi_fluid_k * dd + dw_phi_fluid_k * d_k) * d_test * dx
        elif damage_model_key in {"phase_field", "phase-field", "at2", "energy"}:
            # Energy-derived phase-field damage (AT2-like, lagged drive):
            #   α η_d D_t^S d - div(α G_c l ∇d) + α (G_c/l) d = 2 α (1-d) H_prev,
            # with a *lagged* driving field H_prev (updated between steps for robustness).
            Gc_val = float(damage_Gc)
            ell_val = float(damage_l)
            if Gc_val <= 0.0 or ell_val <= 0.0:
                raise ValueError("damage_model='phase_field' requires damage_Gc>0 and damage_l>0.")
            eta_d_c = _c(float(damage_eta))
            Gc_c = _c(Gc_val)
            ell_c = _c(ell_val)
            Gc_over_l = Gc_c / ell_c
            Gc_l = Gc_c * ell_c

            if damage_H_prev is not None:
                # Optional irreversibility: supply a history field H^{prev}(x)
                # (typically updated as max over time) so damage cannot heal when
                # the instantaneous drive relaxes.
                H_prev = damage_H_prev
            else:
                damage_pf_driver_key = str(damage_pf_driver).strip().lower()
                if damage_pf_driver_key in {"von_mises", "vm", "von-mises"}:
                    # Legacy: scale a von-Mises-based proxy into an energy density.
                    psi0_val = float(damage_psi0)
                    if psi0_val <= 0.0:
                        psi0_val = Gc_val / max(ell_val, 1.0e-12)
                    H_prev = _c(psi0_val) * drive_vm
                elif damage_pf_driver_key in {"miehe", "miehe_energy", "energy", "psi_plus", "psi+"}:
                    eta_pos = float(damage_eta_pos)
                    if solid_model_key in {"linear", "small_strain", "linear_elastic"}:
                        # Miehe-type tensile energy density ψ⁺(u) for linear elasticity:
                        #   ψ⁺ = μ ||ε⁺||² + (λ/2) ⟨tr ε⟩₊².
                        disc_reg = 1.0e-16
                        eps_un = _epsilon(u_n)
                        eps_plus_un, _, _, _, _ = spectral_positive_part_2x2_sym(
                            eps_un, eta_pos=eta_pos, disc_reg=disc_reg
                        )
                        tr_eps = div(u_n)
                        tr_pos = _smooth_pos_u(tr_eps, eta=eta_pos)
                        psi_plus = mu_s * inner(eps_plus_un, eps_plus_un) + _c(0.5) * lambda_s * (tr_pos * tr_pos)
                    elif solid_model_key in {"hencky", "hencky_log", "hencky_log_strain"}:
                        # Finite-strain Hencky tensile energy density ψ⁺(u) in 2D:
                        #   ψ⁺ = μ ||E⁺||² + (λ/2) ⟨tr E⟩₊²,  E=log(V).
                        psi_plus = hencky_tensile_energy_miehe(
                            u_n, mu_s, lambda_s, dim=int(dim), eta_pos=eta_pos
                        )
                    elif solid_model_key in {"stvk", "svk", "saint_venant_kirchhoff", "saint-venant-kirchhoff"}:
                        # Finite-strain SVK tensile energy density ψ⁺(u) in 2D:
                        #   ψ⁺ = μ ||E⁺||² + (λ/2) ⟨tr E⟩₊²,  E=0.5(C-I).
                        psi_plus = svk_tensile_energy_miehe(
                            u_n, mu_s, lambda_s, dim=int(dim), eta_pos=eta_pos
                        )
                    else:
                        raise ValueError(
                            "damage_pf_driver='miehe_energy' requires solid_model in {'linear','stvk','hencky'} (2D). "
                            "Use damage_pf_driver='von_mises' for neo-Hookean."
                        )
                    scale = float(damage_psi0)
                    if scale > 0.0:
                        psi_plus = _c(scale) * psi_plus
                    H_prev = psi_plus
                else:
                    raise ValueError(f"Unknown damage_pf_driver {damage_pf_driver!r}.")

            DtS_d_k = (d_k - d_n) * inv_dt
            DtS_d_k += th * dot(grad(d_k), vS_k) + one_m_th * dot(grad(d_n), vS_n)

            # Consistent with g(d) = (1-κ)(1-d)^2 + κ:
            #   ∂/∂d [ g(d) ψ⁺ ] = -2 (1-κ) (1-d) ψ⁺.
            f_pf_k = eta_d_c * DtS_d_k + Gc_over_l * d_k - _c(2.0) * one_m_kappa_stiff * one_m_d_k * H_prev

            r_damage = alpha_k * d_test * f_pf_k * dx
            r_damage += alpha_k * Gc_l * inner(grad(d_k), grad(d_test)) * dx
            if float(damage_gamma_out) != 0.0:
                r_damage += gamma_out_c * w_phi_fluid_k * d_k * d_test * dx

            d_DtS_d_k = dd * inv_dt + th * (dot(grad(dd), vS_k) + dot(grad(d_k), dvS))
            df_pf_k = eta_d_c * d_DtS_d_k + Gc_over_l * dd + _c(2.0) * one_m_kappa_stiff * H_prev * dd

            a_damage = (dalpha * f_pf_k + alpha_k * df_pf_k) * d_test * dx
            a_damage += (dalpha * Gc_l * inner(grad(d_k), grad(d_test)) + alpha_k * Gc_l * inner(grad(dd), grad(d_test))) * dx
            if float(damage_gamma_out) != 0.0:
                a_damage += gamma_out_c * (w_phi_fluid_k * dd + dw_phi_fluid_k * d_k) * d_test * dx
        else:
            raise ValueError(f"Unknown damage_model {damage_model!r}.")

    # ------------------------------------------------------------------
    # (vi) Substrate transport
    # ------------------------------------------------------------------
    D_S_c = _c(float(D_S))
    CSk = C_k * S_k
    CSn = C_n * S_n

    # Strong: ∂t(CS) + div(CS v) - div(D grad S) + R_S = f_S.
    #
    # IMPORTANT: avoid `dot(CS*v, grad(test))` because the current dot/left_dot
    # implementation cannot handle scalar-test gradients with vector trials.
    # We instead expand div(CS v) = CS div(v) + ∇(CS)·v and use
    # ∇(CS) = S ∇C + C ∇S.
    # Substrate sink R_S:
    # - With S normalized by ρ_s*: R_S = (1/Y) (Π_b/ρ_s*).
    # - With S as mass concentration: multiply by ρ_s* (handled via rho_s_star).
    rho_s_star_c = _c(float(rho_s_star))
    Y_c = _c(float(Y))
    if use_B_primary:
        RS_k = rho_s_star_c * _R_S_consumption_B(
            S_k, B_k, mu_max=_c(float(mu_max)), K_S=_c(float(K_S)), k_d=_c(float(k_d)), Y=Y_c
        )
        RS_n = rho_s_star_c * _R_S_consumption_B(
            S_n, B_n, mu_max=_c(float(mu_max)), K_S=_c(float(K_S)), k_d=_c(float(k_d)), Y=Y_c
        )
    else:
        RS_k = rho_s_star_c * _R_S_consumption(
            S_k, phi_k, alpha_k, mu_max=_c(float(mu_max)), K_S=_c(float(K_S)), k_d=_c(float(k_d)), Y=Y_c
        )
        RS_n = rho_s_star_c * _R_S_consumption(
            S_n, phi_n, alpha_n, mu_max=_c(float(mu_max)), K_S=_c(float(K_S)), k_d=_c(float(k_d)), Y=Y_c
        )

    # IMEX split for stiff substrate reaction:
    # The Monod sink can be very stiff once scaled by ρ_s^*/Y, so Crank–Nicolson
    # (θ=0.5) may produce oscillatory updates unless Δt is extremely small.
    # Allow treating the reaction term fully implicitly while keeping the rest
    # of the transport equation at the global θ-scheme.
    rs_key = str(substrate_reaction_scheme).strip().lower()
    if rs_key in {"theta", "cn", "trap", "trapezoid", "trapezoidal"}:
        th_RS = th
        one_m_th_RS = one_m_th
    elif rs_key in {"implicit", "imex", "be", "backward_euler", "backward-euler", "lstable", "l-stable"}:
        th_RS = _c(1.0)
        one_m_th_RS = _c(0.0)
    elif rs_key in {"explicit", "fe", "forward_euler", "forward-euler"}:
        th_RS = _c(0.0)
        one_m_th_RS = _c(1.0)
    else:
        raise ValueError(
            f"Unknown substrate_reaction_scheme={substrate_reaction_scheme!r}. "
            "Use 'theta' (default), 'implicit'/'imex', or 'explicit'."
        )

    diff_key = str(substrate_diffusion_scheme).strip().lower()
    if diff_key in {"theta", "cn", "trap", "trapezoid", "trapezoidal"}:
        th_diff = th
        one_m_th_diff = one_m_th
    elif diff_key in {"implicit", "imex", "be", "backward_euler", "backward-euler", "lstable", "l-stable"}:
        th_diff = _c(1.0)
        one_m_th_diff = _c(0.0)
    elif diff_key in {"explicit", "fe", "forward_euler", "forward-euler"}:
        th_diff = _c(0.0)
        one_m_th_diff = _c(1.0)
    else:
        raise ValueError(
            f"Unknown substrate_diffusion_scheme={substrate_diffusion_scheme!r}. "
            "Use 'theta' (default), 'implicit'/'imex', or 'explicit'."
        )

    div_CSv_k = CSk * div(v_k) + S_k * dot(gradC_k, v_k) + C_k * dot(grad(S_k), v_k)
    div_CSv_n = CSn * div(v_n) + S_n * dot(gradC_n, v_n) + C_n * dot(grad(S_n), v_n)

    r_sub = S_test * ((CSk - CSn) * inv_dt) * dx
    r_sub += S_test * (th * div_CSv_k + one_m_th * div_CSv_n) * dx
    r_sub += D_S_c * th_diff * inner(grad(S_k), grad(S_test)) * dx + D_S_c * one_m_th_diff * inner(grad(S_n), grad(S_test)) * dx
    r_sub += S_test * (th_RS * RS_k + one_m_th_RS * RS_n) * dx
    r_sub += -S_test * f_S * dx

    # Jacobian (k-part only)
    dCSk = dC * S_k + C_k * dS
    dRS = rho_s_star_c * ((_c(1.0) / Y_c) * dPi)  # RS = (rho_s_star/Y) (Π_b/ρ_s*)

    d_div_CSv_k = dCSk * div(v_k) + CSk * div(dv)
    d_div_CSv_k += dS * dot(gradC_k, v_k) + S_k * dot(dgradC_k, v_k) + S_k * dot(gradC_k, dv)
    d_div_CSv_k += dC_k * dot(grad(S_k), v_k) + C_k * dot(grad(dS), v_k) + C_k * dot(grad(S_k), dv)

    a_sub = S_test * (dCSk * inv_dt) * dx
    a_sub += S_test * th * d_div_CSv_k * dx
    a_sub += D_S_c * th_diff * inner(grad(dS), grad(S_test)) * dx
    a_sub += S_test * th_RS * dRS * dx

    # ------------------------------------------------------------------
    # (vii) Detached biomass transport (optional)
    # ------------------------------------------------------------------
    r_detached = None
    a_detached = None
    if X_k is not None:
        if X_n is None or dX is None or X_test is None:
            raise ValueError("X_k provided but one of (X_n, dX, X_test) is missing.")

        D_X_c = _c(float(D_X))

        CXk = C_k * X_k
        CXn = C_n * X_n

        div_CXv_k = CXk * div(v_k) + X_k * dot(gradC_k, v_k) + C_k * dot(grad(X_k), v_k)
        div_CXv_n = CXn * div(v_n) + X_n * dot(gradC_n, v_n) + C_n * dot(grad(X_n), v_n)

        # Source from detachment: R_det = ρ_s* (1-φ) D_det_prev δ(α).
        #
        # In the support-preserving internal-conversion model there is no
        # erosion from support to suspended biomass, so this interface source is
        # disabled even if an X field is present.
        if support_physics_key in {"internal_conversion", "stored_support"}:
            R_det_k = _c(0.0)
            R_det_n = _c(0.0)
        else:
            R_det_k = rho_s_star_c * _one_minus(phi_k) * D_det_prev * delta_k
            R_det_n = rho_s_star_c * _one_minus(phi_n) * D_det_prev * delta_n

        r_detached = X_test * ((CXk - CXn) * inv_dt) * dx
        r_detached += X_test * (th * div_CXv_k + one_m_th * div_CXv_n) * dx
        r_detached += D_X_c * th * inner(grad(X_k), grad(X_test)) * dx + D_X_c * one_m_th * inner(grad(X_n), grad(X_test)) * dx
        r_detached += -X_test * (th * R_det_k + one_m_th * R_det_n) * dx
        r_detached += -X_test * f_X * dx

        # Jacobian (k-part only)
        dCXk = dC * X_k + C_k * dX

        d_div_CXv_k = dCXk * div(v_k) + CXk * div(dv)
        d_div_CXv_k += dX * dot(gradC_k, v_k) + X_k * dot(dgradC_k, v_k) + X_k * dot(gradC_k, dv)
        d_div_CXv_k += dC * dot(grad(X_k), v_k) + C_k * dot(grad(dX), v_k) + C_k * dot(grad(X_k), dv)

        if support_physics_key in {"internal_conversion", "stored_support"}:
            dR_det_k = _c(0.0)
        else:
            # δR_det_k (D_det_prev is lagged): ρ_s* D_det_prev [ -(δφ) δ(α) + (1-φ) δδ(α) ].
            d_delta_k = _c(4.0) * (_one_minus(_c(2.0) * alpha_k)) * dalpha
            dR_det_k = rho_s_star_c * D_det_prev * ((-dphi) * delta_k + _one_minus(phi_k) * d_delta_k)

        a_detached = X_test * (dCXk * inv_dt) * dx
        a_detached += X_test * th * d_div_CXv_k * dx
        a_detached += D_X_c * th * inner(grad(dX), grad(X_test)) * dx
        a_detached += -X_test * th * dR_det_k * dx

    # ------------------------------------------------------------------
    if float(fluid_momentum_scale_c) != 1.0:
        for key, term in tuple(momentum_terms.items()):
            momentum_terms[key] = term * fluid_momentum_scale_c
        r_mom = r_mom * fluid_momentum_scale_c
        a_mom = a_mom * fluid_momentum_scale_c

    if float(skeleton_momentum_scale_c) != 1.0:
        for key, term in tuple(skeleton_terms.items()):
            skeleton_terms[key] = term * skeleton_momentum_scale_c
        for key, term in tuple(skeleton_jac_terms.items()):
            skeleton_jac_terms[key] = term * skeleton_momentum_scale_c
        r_skeleton = r_skeleton * skeleton_momentum_scale_c
        if r_skeleton_pressure is not None:
            r_skeleton_pressure = r_skeleton_pressure * skeleton_momentum_scale_c
        a_skel = a_skel * skeleton_momentum_scale_c
        if a_skeleton_pressure is not None:
            a_skeleton_pressure = a_skeleton_pressure * skeleton_momentum_scale_c

    residual_form = r_mom + r_mass + r_kinematics + r_skeleton + r_phi + r_alpha + r_sub
    jacobian_form = a_mom + a_mass + a_kinematics + a_skel + a_phi + a_alpha + a_sub
    if r_pore is not None:
        residual_form += r_pore
    if a_pore is not None:
        jacobian_form += a_pore
    if r_B is not None and r_B is not r_phi:
        residual_form += r_B
    if a_B is not None and a_B is not a_phi:
        jacobian_form += a_B
    if r_mu_alpha is not None:
        residual_form += r_mu_alpha
    if a_mu_alpha is not None:
        jacobian_form += a_mu_alpha
    if r_alpha_lambda is not None:
        residual_form += r_alpha_lambda
    if a_alpha_lambda is not None:
        jacobian_form += a_alpha_lambda
    if r_drag_lambda is not None:
        residual_form += r_drag_lambda
    if a_drag_lambda is not None:
        jacobian_form += a_drag_lambda
    if r_damage is not None:
        residual_form += r_damage
    if a_damage is not None:
        jacobian_form += a_damage
    if r_volumetric is not None:
        residual_form += r_volumetric
    if a_volumetric is not None:
        jacobian_form += a_volumetric
    if r_detached is not None:
        residual_form += r_detached
    if a_detached is not None:
        jacobian_form += a_detached

    return BiofilmOneDomainForms(
        jacobian_form=jacobian_form,
        residual_form=residual_form,
        r_momentum=r_mom,
        r_momentum_terms=momentum_terms,
        r_mass=r_mass,
        r_pore=r_pore,
        r_total_mass=r_total_mass,
        r_kinematics=r_kinematics,
        r_kinematics_terms=kinematics_terms,
        r_skeleton=r_skeleton,
        r_skeleton_terms=skeleton_terms,
        r_phi=r_phi,
        r_B=r_B,
        r_alpha=r_alpha,
        r_mu_alpha=r_mu_alpha,
        r_damage=r_damage,
        r_substrate=r_sub,
        a_momentum=a_mom,
        a_mass=a_mass,
        a_pore=a_pore,
        a_total_mass=a_total_mass,
        a_kinematics=a_kinematics,
        a_skeleton=a_skel,
        a_skeleton_terms=skeleton_jac_terms,
        a_phi=a_phi,
        a_B=a_B,
        a_alpha=a_alpha,
        a_mu_alpha=a_mu_alpha,
        a_damage=a_damage,
        a_substrate=a_sub,
        r_detached=r_detached,
        a_detached=a_detached,
        r_alpha_lambda=r_alpha_lambda,
        a_alpha_lambda=a_alpha_lambda,
        r_drag_lambda=r_drag_lambda,
        a_drag_lambda=a_drag_lambda,
        r_skeleton_pressure=r_skeleton_pressure,
        a_skeleton_pressure=a_skeleton_pressure,
        r_volumetric=r_volumetric,
        a_volumetric=a_volumetric,
    )
