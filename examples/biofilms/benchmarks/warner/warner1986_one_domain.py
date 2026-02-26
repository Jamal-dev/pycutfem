"""
Warner & Gujer (1986) benchmark for the *one-domain* diffuse-interface biofilm model.

Scope (by design)
-----------------
The Warner & Gujer model is a 1D (in ζ) multispecies biofilm model (f1,f2,S1,S2,S3,L).
The one-domain model implemented in `examples/utils/biofilm/one_domain.py` is a
diffuse-interface mixture PDE and (currently) uses a *single* substrate field S.

This script therefore performs a *reduced* comparison:
  - Use a 2D rectangular strip with uniform-in-x data to emulate a "1D" setup.
  - Treat the single substrate S as Warner's organics S1 (COD).
  - Compare qualitative features across Warner cases 1--5:
      * growth (case 1),
      * step change in bulk substrate (case 2),
      * shear detachment (case 3),
      * sloughing event (case 4),
      * reactor coupling / bulk limitation (case 5; simplified),
    using thickness and substrate-removal flux time series.

The Warner reference curves are loaded from the finite-difference benchmark outputs
written by `examples/biofilms/benchmarks/warner1986_benchmark.py`.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import (
    NewtonParameters,
    NewtonSolver,
    PdasNewtonSolver,
    TimeStepperParameters,
    VIParameters,
)
from pycutfem.ufl.expressions import Constant, FacetNormal, Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction, dot, grad, inner
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import dS, dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad

from examples.utils.biofilm.adhesion import assemble_scalar
from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms


_TIME_TOL_DAYS = 1.0e-12


def _time_ge(t_days: float, target_days: float) -> bool:
    """Floating-point tolerant `t >= target` comparison in day units."""
    return float(t_days) >= float(target_days) - _TIME_TOL_DAYS


def _time_lt(t_days: float, target_days: float) -> bool:
    """Floating-point tolerant `t < target` comparison in day units."""
    return float(t_days) < float(target_days) - _TIME_TOL_DAYS


def _time_in_half_open_window(t_days: float, start_days: float, end_days: float) -> bool:
    """Floating-point tolerant half-open window check: t ∈ [start, end)."""
    t = float(t_days)
    return _time_ge(t, float(start_days)) and _time_lt(t, float(end_days))


def _tag_rectangle_boundaries(mesh: Mesh, *, L: float, H: float, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - float(L)) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - float(H)) <= tol,
        }
    )


def _apply_y_stretch(nodes: np.ndarray, *, Hy: float, y_stretch: float) -> np.ndarray:
    """
    Optional y-mapping to cluster elements near y=0 and y=Hy (both ends).

    We use a smooth periodic perturbation of the uniform parameter η=y/Hy:
        y_new = Hy * (η + c * sin(2π η)/(2π)),   with |c| < 1.

    For c<0, dy/dη = 1 + c cos(2π η) is smaller near η=0,1 and larger near η=0.5,
    clustering resolution near the two boundaries (useful because the Warner cases
    start with a thin film at y≈0 and later approach y≈Hy).
    """
    c = float(y_stretch)
    if abs(c) <= 1.0e-14:
        return nodes
    if not (-0.99 < c < 0.99):
        raise ValueError(f"y_stretch must be in (-0.99,0.99); got {c}.")
    Hy = float(Hy)
    if Hy <= 0.0:
        return nodes
    # Common case for pycutfem meshes: `nodes` is a list[Node] (see core.topology.Node).
    if isinstance(nodes, (list, tuple)) and len(nodes) > 0 and hasattr(nodes[0], "y"):
        for node in nodes:
            eta = float(node.y) / Hy
            if eta < 0.0:
                eta = 0.0
            elif eta > 1.0:
                eta = 1.0
            node.y = Hy * (eta + c * math.sin(2.0 * math.pi * eta) / (2.0 * math.pi))
        return nodes

    # Fallback: nodes as numeric array-like of shape (n,2) or (n,>=2).
    arr = np.asarray(nodes, dtype=float)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        out = np.array(arr, copy=True)
        eta = np.clip(out[:, 1] / Hy, 0.0, 1.0)
        out[:, 1] = Hy * (eta + c * np.sin(2.0 * math.pi * eta) / (2.0 * math.pi))
        return out

    return nodes


def _mark_inactive_fields(dh: DofHandler, *field_names: str) -> None:
    """Exclude selected fields from the Newton solve via `dh.dof_tags['inactive']`."""
    tags = getattr(dh, "dof_tags", None) or {}
    inactive = set(tags.get("inactive", set()))
    for fname in field_names:
        try:
            sl = np.asarray(dh.get_field_slice(fname), dtype=int).ravel()
        except Exception:
            continue
        inactive.update(int(i) for i in sl)
    tags["inactive"] = inactive
    dh.dof_tags = tags


def _smooth_step(z: np.ndarray) -> np.ndarray:
    # Robust sigmoid: 0.5*(1+tanh(z)).
    return 0.5 * (1.0 + np.tanh(np.asarray(z, dtype=float)))

def _sanitize_run_tag(tag: str) -> str:
    """Convert a user-provided tag into a filesystem-friendly suffix."""
    raw = str(tag or "").strip()
    if not raw:
        return ""
    out = []
    for ch in raw:
        if ch.isspace():
            out.append("_")
            continue
        if ch.isalnum() or ch in {".", "_", "-"}:
            out.append(ch)
    return "".join(out)

def _case2_mu_extra(
    *,
    t_days: float,
    enabled: bool,
    start_days: float,
    mu0: float,
    mu1: float,
    tau_days: float,
) -> float:
    """
    Effective COD-independent volumetric expansion source for case 2.

    Warner case 2 regrows after the COD step due to other substrates/species.
    This reduced one-substrate mapping mimics that by adding an extra net
    expansion rate (1/d) to the mixture constraint only (no COD consumption).
    """
    if not bool(enabled):
        return 0.0
    t = float(t_days)
    t0 = float(start_days)
    if t < t0:
        return 0.0
    mu0 = float(mu0)
    mu1 = float(mu1)
    tau = float(tau_days)
    if not math.isfinite(tau) or tau <= 0.0:
        return float(mu1)
    dt = max(t - t0, 0.0)
    w = 1.0 - math.exp(-dt / tau)
    return float(mu0 + (mu1 - mu0) * w)

def _one_minus(expr):
    # Keep the UFL operand on the left for backend compatibility.
    return (-expr) + Constant(1.0)

def _pow_int(expr, n: int):
    """Integer power helper that stays within the UFL expression algebra."""
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be >= 0; got {n}.")
    if n == 0:
        return Constant(1.0)
    out = expr
    for _ in range(n - 1):
        out = out * expr
    return out


def _alpha_step_eval(x: np.ndarray, y: np.ndarray, *, h: np.ndarray, eps: float) -> np.ndarray:
    """Diffuse Heaviside: alpha≈1 for y<h, alpha≈0 for y>h."""
    eps = max(float(eps), 1.0e-12)
    xx, yy = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    hh = np.broadcast_to(np.asarray(h, dtype=float), xx.shape)
    return _smooth_step((hh - yy) / eps)


def _as_float_time(fn):
    return lambda x, y, t: float(np.asarray(fn(np.asarray(x), np.asarray(y), float(t))).reshape(()))


@dataclass(frozen=True)
class WarnerCODReference:
    t_days: np.ndarray
    L_um: np.ndarray
    removal: np.ndarray  # -jL_1 (positive removal) [g m^-2 d^-1]


def _load_warner_cod_reference(*, case_id: int, warner_outdir: Path) -> WarnerCODReference:
    path = warner_outdir / f"case{int(case_id)}_backend=cpp_timeseries.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Warner1986 reference CSV: {path}\n"
            "Run: `python -u examples/biofilms/benchmarks/warner/warner1986_benchmark.py --case all --backend cpp`"
        )
    data = np.genfromtxt(str(path), delimiter=",", names=True, dtype=float, encoding=None)
    if getattr(data, "shape", ()) == ():
        data = np.array([data], dtype=data.dtype)
    t = np.asarray(data["t_days"], dtype=float)
    L_um = np.asarray(data["L_um"], dtype=float)
    # warner1986_benchmark stores signed jL; removal is -jL
    removal = -np.asarray(data["jL_1"], dtype=float)
    return WarnerCODReference(t_days=t, L_um=L_um, removal=removal)


def _interface_diffusive_flux(
    *,
    dh: DofHandler,
    alpha_k: Function,
    S_k: Function,
    D_S_nondim: float,
    qdeg: int,
    backend: str,
) -> float:
    """
    Approximate the *diffusive* COD flux into the biofilm across the *diffuse* interface.

    Warner defines the net substrate flux at the film-water interface as:
        j_L = j_diff - u_L S_L,     j_diff = -D ∂S/∂n,
    so the plotted "removal" is typically -j_L (positive for uptake).

    For a sharp indicator α, using the diffuse-interface identity with an outward
    normal n_out ≈ -∇α/|∇α| and δ_Γ ≈ |∇α| gives:
        ∫_Γ (-j_diff·n_out) dA  ≈  -∫_Ω D ∇S · ∇α dV.

    This returns the *integral* (over Γ) in nondimensional coordinates. Convert
    to a physical flux (g m^-2 d^-1) by multiplying by L_ref and dividing by the
    strip width.
    """
    D_c = Constant(float(D_S_nondim))
    dx_q = dx(metadata={"q": int(qdeg)})
    I = (-D_c * dot(grad(S_k), grad(alpha_k))) * dx_q
    return float(assemble_scalar(dh, I, backend=backend, quad_order=int(qdeg)))


def _strip_average_by_y(
    *,
    dh: DofHandler,
    field_name: str,
    values: np.ndarray,
    y_round: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(dh.get_dof_coords(field_name), dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    y = coords[:, 1]
    v = np.asarray(values, dtype=float).ravel()
    if y.size != v.size:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    # Group by y-level and average across x.
    y_key = np.round(y, decimals=int(y_round))
    y_levels, inv = np.unique(y_key, return_inverse=True)
    counts = np.bincount(inv)
    sums = np.bincount(inv, weights=v)
    with np.errstate(divide="ignore", invalid="ignore"):
        v_bar = np.where(counts > 0, sums / counts, np.nan)
    order = np.argsort(y_levels)
    return np.asarray(y_levels[order], dtype=float), np.asarray(v_bar[order], dtype=float)


def _strip_interface_metrics(
    *,
    dh: DofHandler,
    alpha: Function,
    S: Function,
    alpha_half: float = 0.5,
    y_round: int = 12,
) -> tuple[float, float, float]:
    """
    Strip-only interface diagnostics from x-averaged profiles.

    Returns
    -------
    L_half : float
        Interface position y(α=alpha_half) in nondimensional coordinates.
    S_surf : float
        Substrate value at y=L_half by linear interpolation (x-averaged).
    dS_dy : float
        Substrate gradient (nondim) across the bracketing y-levels (x-averaged).

    Notes
    -----
    Warner reports both:
      - diffusive flux  j_diff = -D ∂S/∂n  at the interface,
      - total flux      j_L    = j_diff - u_L S_L.
    Their plotted "removal" is -j_L (positive for uptake). For strip runs we can
    approximate (-j_diff) and (-j_L) using (L_half, S_surf, dS_dy, u_L).
    """
    y_a, a_bar = _strip_average_by_y(
        dh=dh,
        field_name="alpha",
        values=np.asarray(alpha.nodal_values, dtype=float),
        y_round=int(y_round),
    )
    if y_a.size < 2:
        return 0.0, float("nan"), float("nan")

    mask = np.isfinite(a_bar) & (a_bar >= float(alpha_half))
    if not np.any(mask):
        return 0.0, float("nan"), float("nan")
    j = int(np.max(np.nonzero(mask)[0]))
    if j >= y_a.size - 1:
        return float(y_a[-1]), float("nan"), float("nan")

    y0 = float(y_a[j])
    y1 = float(y_a[j + 1])
    a0 = float(a_bar[j])
    a1 = float(a_bar[j + 1])
    if not (math.isfinite(a0) and math.isfinite(a1)) or y1 <= y0:
        return float(y0), float("nan"), float("nan")
    if abs(a1 - a0) <= 1.0e-16:
        y_half = float(y0)
    else:
        y_half = float(y0 + (float(alpha_half) - a0) * (y1 - y0) / (a1 - a0))

    y_s, S_bar = _strip_average_by_y(
        dh=dh,
        field_name="S",
        values=np.asarray(S.nodal_values, dtype=float),
        y_round=int(y_round),
    )
    if y_s.size < 2:
        return float(y_half), float("nan"), float("nan")
    if y_s.shape != y_a.shape or not np.allclose(y_s, y_a, atol=0.0, rtol=0.0):
        S_bar = np.interp(y_a, y_s, S_bar)

    S0 = float(S_bar[j])
    S1 = float(S_bar[j + 1])
    dy = float(y1 - y0)
    if dy <= 0.0:
        return float(y_half), float("nan"), float("nan")
    dS_dy = float((S1 - S0) / dy)
    t = float((y_half - y0) / dy)
    S_surf = float(S0 + t * (S1 - S0))
    return float(y_half), float(S_surf), float(dS_dy)


def _S_bulk_case(case_id: int, t_days: float, *, S_high: float) -> float:
    if int(case_id) == 2 and _time_ge(float(t_days), 6.0):
        return 0.0
    return float(S_high)


def _detachment_coeff_case(
    *,
    case_id: int,
    t_days: float,
    dt_days: float,
    L_thickness_nondim: float,
    L_ref_m: float,
    eps_det_nondim: float,
    lambda_shear: float,
    slough_mode: str,
    slough_drop_nondim: float,
) -> float:
    """
    Return D_det_prev (units 1/day) used in the alpha equation as:
        RHS detachment sink = -D_det_prev * δ(α)

    For case 3 we mimic Warner's shear law σ = -λ L^2 via:
        V_det = λ L_phys^2  [m/d]
        V_det_nondim = V_det / L_ref
        D_det_prev ≈ V_det_nondim / (4 eps_det)

    For case 4 we mimic the short sloughing event σ=-0.05 m/d on (5.984,5.994) d
    when `slough_mode='exact'`. Other modes may apply sloughing by different means.
    """
    cid = int(case_id)
    t = float(t_days)
    eps = max(float(eps_det_nondim), 1.0e-12)
    L = max(float(L_thickness_nondim), 0.0)
    slough_key = str(slough_mode).strip().lower()

    if cid == 3:
        # Warner: sigma = -lambda * L^2 with lambda=750 m^{-1} d^{-1}.
        # Use V_det = +lambda*L^2 for recession speed magnitude.
        # Convert to nondimensional length/time with L_phys=L_ref*L_nondim:
        #   V_det_nondim = (lambda * (L_ref*L)^2) / L_ref = (lambda*L_ref) * L^2.
        V_det_nondim = float(lambda_shear) * float(L_ref_m) * (L * L)
        return float(V_det_nondim / (4.0 * eps))

    if cid == 4 and slough_key == "exact" and _time_in_half_open_window(t, 5.984, 5.994):
        # Sloughing speed magnitude in Warner: 0.05 m/d.
        V_slough_nondim = 0.05 / float(L_ref_m)
        return float(V_slough_nondim / (4.0 * eps))

    if cid == 4 and slough_key in {"integrated", "coarse"}:
        # Coarse sloughing: apply a single-step thickness drop around t=6d.
        dt = max(float(dt_days), 1.0e-16)
        t0 = float(t)
        t1 = float(t0 + dt)
        if float(t0) < 6.0 - _TIME_TOL_DAYS and _time_ge(float(t1), 6.0):
            V_slough_nondim = float(slough_drop_nondim) / dt
            return float(V_slough_nondim / (4.0 * eps))

    return 0.0


def _apply_slough_truncate_strip(
    *,
    dh: DofHandler,
    alpha: Function,
    drop_nondim: float,
    eps_nondim: float,
    x_round: int = 12,
) -> None:
    """
    Case-4 sloughing "truncate" update for strip-like runs.

    We remove a thickness `drop_nondim` from the *top* of the biofilm while keeping the
    bottom fixed, i.e. we enforce a new interface position
        L_new = L_old - drop
    by applying an additional diffuse cutoff at y=L_new:
        alpha_new(x,y) = min(alpha_old(x,y), H_eps(L_new - y)).

    This is a closer analogue of Warner's sloughing (surface recession) than a pure
    vertical translation of α, which would incorrectly shift the entire profile.
    """
    drop = float(drop_nondim)
    if not (math.isfinite(drop) and drop > 0.0):
        return
    eps = max(float(eps_nondim), 1.0e-12)
    coords = np.asarray(dh.get_dof_coords("alpha"), dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        return
    a0 = np.asarray(alpha.nodal_values, dtype=float).ravel()
    if a0.size != coords.shape[0]:
        return

    try:
        L_old = float(_strip_thickness_alpha_half(dh=dh, alpha=alpha))
    except Exception:
        # Fallback for pathological profiles: assume the current "thickness" scale is the max y where α is significant.
        y = np.asarray(coords[:, 1], dtype=float)
        mask = np.asarray(a0, dtype=float) >= 0.5
        L_old = float(np.max(y[mask])) if np.any(mask) else 0.0

    L_new = float(max(L_old - drop, 0.0))

    # Cutoff profile H_eps(L_new - y).
    x = coords[:, 0]
    y = coords[:, 1]
    x_key = np.round(x, decimals=int(x_round))
    x_levels, inv = np.unique(x_key, return_inverse=True)

    out = np.array(a0, copy=True)
    for xi in range(int(x_levels.size)):
        idx = np.nonzero(inv == xi)[0]
        if idx.size == 0:
            continue
        yy = np.asarray(y[idx], dtype=float)
        aa = np.asarray(a0[idx], dtype=float)
        cutoff = _smooth_step((L_new - yy) / eps)
        out[idx] = np.minimum(aa, cutoff)

    alpha.nodal_values[:] = np.clip(out, 0.0, 1.0)


def _apply_slough_shift_strip_scalar(
    *,
    dh: DofHandler,
    f: Function,
    field: str,
    drop_nondim: float,
    x_round: int = 12,
    clip_min: float | None = None,
) -> None:
    """Generic strip shift: f_new(x,y) = f_old(x, y + drop)."""
    drop = float(drop_nondim)
    if not (math.isfinite(drop) and drop > 0.0):
        return
    coords = np.asarray(dh.get_dof_coords(field), dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        return
    a0 = np.asarray(f.nodal_values, dtype=float).ravel()
    if a0.size != coords.shape[0]:
        return

    x = coords[:, 0]
    y = coords[:, 1]
    x_key = np.round(x, decimals=int(x_round))
    x_levels, inv = np.unique(x_key, return_inverse=True)

    out = np.array(a0, copy=True)
    for xi in range(int(x_levels.size)):
        idx = np.nonzero(inv == xi)[0]
        if idx.size == 0:
            continue
        yy = np.asarray(y[idx], dtype=float)
        aa = np.asarray(a0[idx], dtype=float)
        order = np.argsort(yy)
        yy = yy[order]
        aa = aa[order]
        if yy.size < 2:
            continue
        yq = yy + drop
        out[idx[order]] = np.interp(yq, yy, aa, left=float(aa[0]), right=float(aa[-1]))

    if clip_min is not None:
        out = np.maximum(out, float(clip_min))
    f.nodal_values[:] = out


def _apply_case4_shift_slough_strip(
    *,
    t_days: float,
    applied: bool,
    dh: DofHandler,
    alpha: Function,
    S: Function,
    drop_nondim: float,
    eps_nondim: float,
    S_bulk_value: float,
) -> bool:
    """
    Apply case-4 instantaneous sloughing for strip-like runs ("shift" mode).

    We apply the truncation-from-top update once at t>=5.994d (end of Warner's
    window) and refill the new fluid region above the shifted interface with
    bulk substrate so diagnostics at the event time are consistent.

    Returns the updated `applied` flag.
    """
    if bool(applied):
        return True
    if not _time_ge(float(t_days), 5.994):
        return False
    _apply_slough_truncate_strip(
        dh=dh,
        alpha=alpha,
        drop_nondim=float(drop_nondim),
        eps_nondim=float(eps_nondim),
    )
    try:
        L_new = float(_strip_thickness_alpha_half(dh=dh, alpha=alpha))
        coords_S = np.asarray(dh.get_dof_coords("S"), dtype=float)
        if coords_S.ndim == 2 and coords_S.shape[1] >= 2 and coords_S.shape[0] == np.asarray(S.nodal_values).size:
            mask = coords_S[:, 1] >= float(L_new)
            S.nodal_values[:] = np.where(mask, float(S_bulk_value), S.nodal_values)
    except Exception:
        pass
    return True


def _build_one_domain_problem(
    *,
    Lx: float,
    Hy: float,
    nx: int,
    ny: int,
    y_stretch: float,
    qdeg: int,
    dt_days: float,
    theta: float,
    backend: str,
    mechanics_mode: str,
    freeze_alpha: bool,
    freeze_u: bool,
    freeze_vSx: bool,
    freeze_phi: bool,
    s_v_mode: str,
    s_v_lagged: bool,
    s_v_jacobian: str,
    gamma_u: float,
    u_extension_mode: str,
    gamma_u_pin: float,
    # alpha regularization (optional)
    alpha_cahn_M: float,
    alpha_cahn_gamma: float,
    alpha_cahn_eps: float,
    alpha_cahn_mobility: str,
    alpha_cahn_conservative: bool,
    alpha_cahn_conservative_mode: str,
    # initial conditions
    h0: float,
    eps0: float,
    phi_b: float,
    S0: float,
    init_vS_profile: str,
    # model parameters (single substrate)
    D_S: float,
    substrate_reaction_scheme: str,
    substrate_diffusion_scheme: str,
    D_alpha: float,
    alpha_advection_form: str,
    D_phi: float,
    gamma_phi: float,
    rho_f: float,
    mu_f: float,
    kappa_inv: float,
    mu_s: float,
    lambda_s: float,
    solid_visco_eta: float,
    mu_max: float,
    K_S: float,
    k_g,
    k_d: float,
    Y: float,
    rho_s_star: float,
    # detachment coefficient (lagged Constant)
    D_det_prev: Constant,
):
    nodes, elems, _, corners = structured_quad(Lx, Hy, nx=int(nx), ny=int(ny), poly_order=2)
    nodes = _apply_y_stretch(nodes, Hy=float(Hy), y_stretch=float(y_stretch))
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    _tag_rectangle_boundaries(mesh, L=Lx, H=Hy)

    alpha_cahn_conservative = bool(alpha_cahn_conservative)
    cons_mode = str(alpha_cahn_conservative_mode).strip().lower()
    if cons_mode in {"unknown", "solve"}:
        cons_mode = "unknown"
    elif cons_mode in {"eliminate", "elim", "project", "projected"}:
        cons_mode = "eliminate"
    else:
        raise ValueError(
            f"Unknown alpha_cahn_conservative_mode={alpha_cahn_conservative_mode!r}. Use 'unknown' or 'eliminate'."
        )
    solve_lambda = alpha_cahn_conservative and cons_mode == "unknown"

    field_specs: dict[str, object] = {
        "v_x": 2,
        "v_y": 2,
        "p": 1,
        "vS_x": 2,
        "vS_y": 2,
        "u_x": 2,
        "u_y": 2,
        "phi": 1,
        "alpha": 1,
        **({"lambda_alpha": ":number:"} if alpha_cahn_conservative else {}),
        "S": 1,
    }
    me = MixedElement(mesh, field_specs=field_specs)
    dh = DofHandler(me, method="cg")
    mech = str(mechanics_mode).strip().lower()
    if mech not in {"frozen", "skeleton", "full"}:
        raise ValueError(f"Unknown mechanics_mode={mechanics_mode!r}. Use 'frozen', 'skeleton', or 'full'.")
    if mech == "frozen":
        # Only solve transport + interface (phi, alpha, S).
        _mark_inactive_fields(dh, "v_x", "v_y", "p", "vS_x", "vS_y", "u_x", "u_y")
    elif mech == "skeleton":
        # Keep the mixture velocity v frozen at 0; solve (p, vS, u, phi, alpha, S).
        _mark_inactive_fields(dh, "v_x", "v_y")
    if bool(freeze_phi):
        _mark_inactive_fields(dh, "phi")
    if bool(freeze_alpha):
        _mark_inactive_fields(dh, "alpha")
    if bool(freeze_u):
        _mark_inactive_fields(dh, "u_x", "u_y")
    if bool(freeze_vSx):
        _mark_inactive_fields(dh, "vS_x")
    if alpha_cahn_conservative and (not solve_lambda):
        _mark_inactive_fields(dh, "lambda_alpha")

    V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
    VS = FunctionSpace("VS", ["vS_x", "vS_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    dv = VectorTrialFunction(space=V, dof_handler=dh)
    dvS = VectorTrialFunction(space=VS, dof_handler=dh)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    dp = TrialFunction("p", dof_handler=dh)
    dphi = TrialFunction("phi", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    dlambda_alpha = TrialFunction("lambda_alpha", dof_handler=dh) if solve_lambda else None
    dS = TrialFunction("S", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    vS_test = VectorTestFunction(space=VS, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    lambda_alpha_test = TestFunction("lambda_alpha", dof_handler=dh) if solve_lambda else None
    S_test = TestFunction("S", dof_handler=dh)

    # Unknowns (k) and previous state (n)
    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    vS_k = VectorFunction("vS_k", ["vS_x", "vS_y"], dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    lambda_alpha_k = Function("lambda_alpha_k", "lambda_alpha", dof_handler=dh) if alpha_cahn_conservative else None
    S_k = Function("S_k", "S", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    vS_n = VectorFunction("vS_n", ["vS_x", "vS_y"], dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    lambda_alpha_n = Function("lambda_alpha_n", "lambda_alpha", dof_handler=dh) if alpha_cahn_conservative else None
    S_n = Function("S_n", "S", dof_handler=dh)

    # Initial fields: static strip (v=vS=u=0, p=0), diffuse biofilm layer at the bottom.
    for vf in (v_k, vS_k, u_k, v_n, vS_n, u_n):
        vf.nodal_values[:] = 0.0
    for sf in (p_k, p_n):
        sf.nodal_values[:] = 0.0

    def alpha0(x, y):
        return _alpha_step_eval(x, y, h=float(h0), eps=float(eps0))

    alpha_n.set_values_from_function(alpha0)
    alpha_k.nodal_values[:] = alpha_n.nodal_values[:]

    # IMPORTANT: φ is the *biofilm* porosity used in the mixture interpolation
    #   C = (1-α) + α φ,  B = α (1-φ).
    # Therefore we must NOT pre-blend φ with α (which would double-count the
    # mixture transition and effectively replace B by ~α²(1-φ_b), reducing
    # growth/consumption by ~50% for a tanh-like interface).
    #
    # For the Warner benchmark we typically freeze φ to a constant biofilm
    # porosity φ_b and let α handle the region transition.
    phi_n.set_values_from_function(lambda x, y: float(phi_b))
    phi_k.nodal_values[:] = phi_n.nodal_values[:]

    S_n.set_values_from_function(lambda x, y: float(S0))
    S_k.nodal_values[:] = S_n.nodal_values[:]
    if lambda_alpha_k is not None and lambda_alpha_n is not None:
        lambda_alpha_n.nodal_values[:] = 0.0
        lambda_alpha_k.nodal_values[:] = lambda_alpha_n.nodal_values[:]

    init_key = str(init_vS_profile).strip().lower()
    if init_key in {"growth", "sv", "source"}:
        coords = np.asarray(dh.get_dof_coords("vS_y"), dtype=float)
        if coords.ndim == 2 and coords.shape[1] >= 2:
            x = coords[:, 0]
            y = coords[:, 1]
            a0 = _alpha_step_eval(x, y, h=float(h0), eps=float(eps0))
            S0_v = float(S0)
            mu_net = float(mu_max) * (S0_v / (S0_v + float(K_S))) - float(k_d)
            vSy_guess = float(mu_net) * y * a0
            sl_y = np.asarray(dh.get_field_slice("vS_y"), dtype=int).ravel()
            vS_n.set_nodal_values(sl_y, vSy_guess)
            vS_k.set_nodal_values(sl_y, vSy_guess)
    elif init_key in {"zero", "0", "none"}:
        pass
    else:
        raise ValueError(f"Unknown init_vS_profile={init_vS_profile!r}. Use 'growth' or 'zero'.")

    dt_c = Constant(float(dt_days))
    mu_extra_c = Constant(0.0)

    # Volume source in the mixture constraint: div(C v + B vS) = alpha*s_v.
    # For growth-driven expansion we use the same Pi_b/rho_s* structure as in the phi equation:
    #   Pi_b/rho_s* = (monod(S) - k_d) (1-phi) alpha
    # so set s_v = (monod(S) - k_d) (1-phi), letting the prefactor alpha be applied in the form.
    s_v_key = str(s_v_mode).strip().lower()
    jac_key = str(s_v_jacobian).strip().lower()
    if jac_key in {"frozen", "lagged", "picard"}:
        jac_key = "frozen"
    elif jac_key in {"full", "newton", "consistent"}:
        jac_key = "full"
    else:
        raise ValueError(f"Unknown s_v_jacobian={s_v_jacobian!r}. Use 'full' or 'frozen'.")
    if s_v_key in {"none", "0", "zero"}:
        s_v = Constant(0.0)
        ds_v = Constant(0.0)
    elif s_v_key in {"mu", "mu_net", "munet", "rate"}:
        # Use the net specific growth rate as the volumetric expansion source:
        #   div(C v + B vS) = alpha * (mu_net),
        # where mu_net = monod(S) - k_d.
        #
        # This choice matches the sharp-interface early-time regime when v≈vS and φ is (approximately) constant:
        #   div(vS) ≈ mu_net  =>  L(t) ≈ L0 exp(mu_net t).
        #
        # If instead v is frozen to 0 (mechanics="skeleton"), note that
        #   div(C v + B vS) = div(B vS),
        # so using s_v="pi" may be more appropriate in that case.
        mu_max_c = Constant(float(mu_max))
        K_S_c = Constant(float(K_S))
        k_d_c = Constant(float(k_d))
        if bool(s_v_lagged):
            monod = mu_max_c * (S_n / (S_n + K_S_c))
            s_v = monod - k_d_c + mu_extra_c
            ds_v = Constant(0.0)
        else:
            monod = mu_max_c * (S_k / (S_k + K_S_c))
            s_v = monod - k_d_c + mu_extra_c
            if jac_key == "full":
                denom = S_k + K_S_c
                dmonod = mu_max_c * (K_S_c / (denom * denom)) * dS
                ds_v = dmonod
            else:
                ds_v = Constant(0.0)
    elif s_v_key in {"pi", "pib", "growth"}:
        mu_max_c = Constant(float(mu_max))
        K_S_c = Constant(float(K_S))
        k_d_c = Constant(float(k_d))
        if bool(s_v_lagged):
            monod = mu_max_c * (S_n / (S_n + K_S_c))
            s_v = (monod - k_d_c + mu_extra_c) * _one_minus(phi_n)
            ds_v = Constant(0.0)
        else:
            monod = mu_max_c * (S_k / (S_k + K_S_c))
            s_v = (monod - k_d_c + mu_extra_c) * _one_minus(phi_k)
            if jac_key == "full":
                denom = S_k + K_S_c
                dmonod = mu_max_c * (K_S_c / (denom * denom)) * dS
                ds_v = dmonod * _one_minus(phi_k) - (monod - k_d_c + mu_extra_c) * dphi
            else:
                ds_v = Constant(0.0)
    else:
        raise ValueError(f"Unknown s_v_mode={s_v_mode!r}. Use 'none', 'mu', or 'pi'.")

    forms = build_biofilm_one_domain_forms(
        v_k=v_k,
        p_k=p_k,
        vS_k=vS_k,
        u_k=u_k,
        phi_k=phi_k,
        alpha_k=alpha_k,
        S_k=S_k,
        v_n=v_n,
        p_n=p_n,
        vS_n=vS_n,
        u_n=u_n,
        phi_n=phi_n,
        alpha_n=alpha_n,
        S_n=S_n,
        dv=dv,
        dp=dp,
        dvS=dvS,
        du=du,
        dphi=dphi,
        dalpha=dalpha,
        dS=dS,
        v_test=v_test,
        q_test=q_test,
        vS_test=vS_test,
        u_test=u_test,
        phi_test=phi_test,
        alpha_test=alpha_test,
        S_test=S_test,
        dx=dx(metadata={"q": int(qdeg)}),
        dt=dt_c,
        theta=float(theta),
        rho_f=Constant(float(rho_f)),
        mu_f=Constant(float(mu_f)),
        kappa_inv=Constant(float(kappa_inv)),
        mu_s=Constant(float(mu_s)),
        lambda_s=Constant(float(lambda_s)),
        solid_visco_eta=float(solid_visco_eta),
        gamma_u=float(gamma_u),
        u_extension_mode=str(u_extension_mode),
        gamma_u_pin=float(gamma_u_pin),
        D_phi=float(D_phi),
        gamma_phi=float(gamma_phi),
        D_alpha=float(D_alpha),
        alpha_advection_form=str(alpha_advection_form),
        alpha_cahn_M=float(alpha_cahn_M),
        alpha_cahn_gamma=float(alpha_cahn_gamma),
        alpha_cahn_eps=float(alpha_cahn_eps),
        alpha_cahn_mobility=str(alpha_cahn_mobility),
        alpha_cahn_conservative=bool(alpha_cahn_conservative),
        alpha_cahn_conservative_mode=str(cons_mode),
        lambda_alpha_k=lambda_alpha_k,
        lambda_alpha_n=lambda_alpha_n,
        dlambda_alpha=dlambda_alpha,
        lambda_alpha_test=lambda_alpha_test,
        D_S=float(D_S),
        substrate_reaction_scheme=str(substrate_reaction_scheme),
        substrate_diffusion_scheme=str(substrate_diffusion_scheme),
        mu_max=float(mu_max),
        K_S=float(K_S),
        k_g=k_g,
        k_d=float(k_d),
        Y=float(Y),
        rho_s_star=float(rho_s_star),
        k_det=0.0,
        D_det_prev=D_det_prev,
        s_v=s_v,
        ds_v=ds_v,
    )

    # BCs:
    # - frozen: clamp all mechanics to 0 everywhere.
    # - skeleton: keep v=0; anchor solid (vS,u) on the bottom and remove lateral motion.
    bcs: list[BoundaryCondition] = []
    if mech == "frozen":
        for tag in ("left", "right", "bottom", "top"):
            bcs.extend(
                [
                    BoundaryCondition("v_x", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)),
                    BoundaryCondition("v_y", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)),
                    BoundaryCondition("p", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)),
                    BoundaryCondition("vS_x", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)),
                    BoundaryCondition("vS_y", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)),
                    BoundaryCondition("u_x", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)),
                    BoundaryCondition("u_y", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)),
                ]
            )
    elif mech == "skeleton":
        # Freeze mixture velocity v to 0 (no imposed bulk flow for the strip benchmark).
        for tag in ("left", "right", "bottom", "top"):
            bcs.extend(
                [
                    BoundaryCondition("v_x", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)),
                    BoundaryCondition("v_y", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)),
                ]
            )
        # Fix pressure gauge (otherwise p is determined up to a constant when v is fixed).
        bcs.append(BoundaryCondition("p", "dirichlet", "top", _as_float_time(lambda x, y, t: 0.0)))

        # Anchor skeleton at the support and remove lateral motion (strip symmetry).
        for tag in ("left", "right"):
            bcs.append(BoundaryCondition("vS_x", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)))
            bcs.append(BoundaryCondition("u_x", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)))
        for tag in ("bottom",):
            bcs.append(BoundaryCondition("vS_x", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)))
            bcs.append(BoundaryCondition("vS_y", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)))
            bcs.append(BoundaryCondition("u_x", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)))
            bcs.append(BoundaryCondition("u_y", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)))
    else:
        # Full mechanics: allow vertical mixture flow to accommodate growth (open top).
        # - bottom: no-slip (v=0), clamp skeleton (u=0, vS=0).
        # - left/right: symmetry-like (no horizontal motion).
        # - top: pressure gauge p=0, no horizontal motion.
        bcs.append(BoundaryCondition("v_x", "dirichlet", "bottom", _as_float_time(lambda x, y, t: 0.0)))
        bcs.append(BoundaryCondition("v_y", "dirichlet", "bottom", _as_float_time(lambda x, y, t: 0.0)))
        for tag in ("left", "right", "top"):
            bcs.append(BoundaryCondition("v_x", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)))

        bcs.append(BoundaryCondition("p", "dirichlet", "top", _as_float_time(lambda x, y, t: 0.0)))

        for tag in ("left", "right"):
            bcs.append(BoundaryCondition("vS_x", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)))
            bcs.append(BoundaryCondition("u_x", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)))
        for tag in ("bottom",):
            bcs.append(BoundaryCondition("vS_x", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)))
            bcs.append(BoundaryCondition("vS_y", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)))
            bcs.append(BoundaryCondition("u_x", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)))
            bcs.append(BoundaryCondition("u_y", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)))

    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, (lambda x, y: 0.0)) for b in bcs]

    fields = {"v": v_k, "p": p_k, "vS": vS_k, "u": u_k, "phi": phi_k, "alpha": alpha_k, "S": S_k}
    prev_fields = {"v": v_n, "p": p_n, "vS": vS_n, "u": u_n, "phi": phi_n, "alpha": alpha_n, "S": S_n}
    if lambda_alpha_k is not None and lambda_alpha_n is not None:
        fields["lambda_alpha"] = lambda_alpha_k
        prev_fields["lambda_alpha"] = lambda_alpha_n

    functions = [v_k, p_k, vS_k, u_k, phi_k, alpha_k, S_k]
    prev_functions = [v_n, p_n, vS_n, u_n, phi_n, alpha_n, S_n]
    if lambda_alpha_k is not None and lambda_alpha_n is not None:
        functions.append(lambda_alpha_k)
        prev_functions.append(lambda_alpha_n)

    return {
        "mesh": mesh,
        "me": me,
        "dh": dh,
        "dt_c": dt_c,
        "mu_extra_c": mu_extra_c,
        "forms": forms,
        "bcs": bcs,
        "bcs_homog": bcs_homog,
        "functions": functions,
        "prev_functions": prev_functions,
        "fields": fields,
        "prev_fields": prev_fields,
        "S_k": S_k,
        "alpha_k": alpha_k,
        "phi_k": phi_k,
        "S_n": S_n,
        "alpha_n": alpha_n,
        "phi_n": phi_n,
        "lambda_alpha_k": lambda_alpha_k,
        "lambda_alpha_n": lambda_alpha_n,
    }

def _strip_thickness_alpha_half(
    *,
    dh: DofHandler,
    alpha: Function,
    y_round: int = 12,
    alpha_half: float = 0.5,
) -> float:
    """
    Geometric thickness estimate for a 1D-like strip by locating the height where
    the x-averaged alpha crosses `alpha_half` (default 0.5).

    Returns thickness in *nondimensional* units.
    """
    coords = np.asarray(dh.get_dof_coords("alpha"), dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        return float("nan")
    y = coords[:, 1]
    a = np.asarray(alpha.nodal_values, dtype=float).ravel()
    if y.size != a.size:
        return float("nan")

    # Group by y-level and average alpha across x.
    y_key = np.round(y, decimals=int(y_round))
    y_levels, inv = np.unique(y_key, return_inverse=True)
    counts = np.bincount(inv)
    sums = np.bincount(inv, weights=a)
    with np.errstate(divide="ignore", invalid="ignore"):
        a_bar = np.where(counts > 0, sums / counts, np.nan)

    # Sort by y increasing (unique already sorted, but keep explicit).
    order = np.argsort(y_levels)
    y_levels = y_levels[order]
    a_bar = a_bar[order]

    # Find the highest y where a_bar >= alpha_half, then interpolate to the next node.
    mask = np.isfinite(a_bar) & (a_bar >= float(alpha_half))
    if not np.any(mask):
        return 0.0
    j = int(np.max(np.nonzero(mask)[0]))
    if j >= y_levels.size - 1:
        return float(y_levels[-1])
    y0 = float(y_levels[j])
    y1 = float(y_levels[j + 1])
    a0 = float(a_bar[j])
    a1 = float(a_bar[j + 1])
    if not (math.isfinite(a0) and math.isfinite(a1)) or y1 <= y0:
        return float(y0)
    if abs(a1 - a0) <= 1.0e-16:
        return float(y0)
    return float(y0 + (float(alpha_half) - a0) * (y1 - y0) / (a1 - a0))


def main() -> None:
    ap = argparse.ArgumentParser(description="One-domain reduced Warner1986 benchmark (single substrate, strip + optional 2D).")
    ap.add_argument("--case", type=str, default="all", choices=("all", "1", "2", "3", "4", "5"))
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--mode", type=str, default="strip", choices=("strip", "2d", "both"))
    ap.add_argument(
        "--cases-2d",
        type=str,
        default="1",
        help="Comma-separated list of case ids to run in 2D when --mode is '2d' or 'both' (default: '1').",
    )
    ap.add_argument(
        "--mechanics",
        type=str,
        default="skeleton",
        choices=("frozen", "skeleton", "full"),
        help="Which sub-blocks of the one-domain model to solve (see script header for intent).",
    )
    ap.add_argument(
        "--solve-strategy",
        type=str,
        default="monolithic",
        choices=("monolithic", "split"),
        help=(
            "Nonlinear solve strategy. "
            "'monolithic' solves all active fields in one Newton/VI solve per time step. "
            "'split' performs two solves per step: (mechanics/interface) with frozen S, then (substrate) with frozen others."
        ),
    )
    ap.add_argument(
        "--freeze-u",
        action="store_true",
        help="Hold u fixed at its initial value (useful when mu_s=lambda_s=0; removes the kinematic block).",
    )
    ap.add_argument(
        "--freeze-alpha",
        action="store_true",
        help="Hold alpha fixed at its initial profile (debugging / block isolation).",
    )
    ap.add_argument(
        "--freeze-vSx",
        action="store_true",
        help="Hold vS_x fixed at its initial value (strip: enforces purely vertical skeleton motion).",
    )
    ap.add_argument(
        "--init-vS",
        type=str,
        default="growth",
        choices=("growth", "zero"),
        help="Initial guess for vS (strip): 'growth' seeds vS_y with div(vS)≈mu_net; 'zero' sets vS=0.",
    )
    ap.add_argument(
        "--s-v-mode",
        type=str,
        default="auto",
        choices=("auto", "none", "mu", "pi"),
        help=(
            "Volume source s_v in div(C v + B vS) = alpha*s_v. "
            "'auto' picks 'mu' for mechanics='full' and 'pi' otherwise. "
            "'mu' uses (monod(S)-k_d). "
            "'pi' uses (monod(S)-k_d)*(1-phi)."
        ),
    )
    ap.add_argument(
        "--s-v-lagged",
        action="store_true",
        help="Use lagged (previous-step) fields in s_v (more robust for monolithic Newton).",
    )
    ap.add_argument(
        "--s-v-jacobian",
        type=str,
        default="full",
        choices=("full", "frozen"),
        help="Jacobian of s_v in the volume constraint: 'full' includes δs_v when s_v depends on (S,phi); 'frozen' sets δs_v=0 (Picard).",
    )
    ap.add_argument("--freeze-phi", action="store_true", help="Hold phi fixed to its initial profile (closer to Warner's constant-density assumption).")
    ap.add_argument("--outdir", type=str, default="examples/biofilms/results/warner1986_one_domain")
    ap.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional suffix tag appended to output filenames (useful for parameter/mesh sweeps).",
    )
    ap.add_argument("--paper-figdir", type=str, default="", help="Optional directory to write LaTeX-ready PDF figures.")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--diagnostics", action="store_true", help="Print min/max diagnostics for (alpha,phi,S) each step.")
    ap.add_argument(
        "--bulk-mode",
        type=str,
        default="well_mixed",
        choices=("top", "well_mixed"),
        help="Bulk-substrate modeling: 'top' uses only the top Dirichlet BC; 'well_mixed' relaxes S->S_bulk in fluid (alpha≈0) to emulate fixed surface concentration (Warner cases 1–4).",
    )
    ap.add_argument(
        "--bulk-gamma",
        type=float,
        default=1.0e3,
        help="Relaxation rate (1/d) used with --bulk-mode well_mixed.",
    )
    ap.add_argument(
        "--bulk-alpha-power",
        type=int,
        default=16,
        help=(
            "Exponent m in the bulk-relaxation weight w(α)=(1-α)^m used with --bulk-mode well_mixed. "
            "Using a high power confines the relaxation to the bulk fluid (α≈0) and avoids directly "
            "injecting substrate in the diffuse interface."
        ),
    )
    ap.add_argument(
        "--bulk-jacobian",
        type=str,
        default="full",
        choices=("full", "frozen"),
        help=(
            "Linearization mode for the bulk-relaxation term used with --bulk-mode well_mixed. "
            "'full' includes the α-dependence of w(α) in the Jacobian; "
            "'frozen' treats w(α) as fixed (Picard), which can improve robustness when "
            "using small --bulk-alpha-power."
        ),
    )
    ap.add_argument(
        "--removal-metric",
        type=str,
        default="interface_total",
        choices=("interface_total", "interface_diff", "top_flux", "bulk_exchange", "reservoir_exchange", "consumption"),
        help=(
            "How to compute substrate removal (for comparison curves). "
            "'interface_total' is closest to Warner's definition (-j_L = -j_diff + u_L S_L). "
            "'reservoir_exchange' measures net transfer from the imposed bulk (well_mixed relaxation) plus the top-boundary flux "
            "(and includes the interface penalty exchange if --surface-gamma is used)."
        ),
    )
    ap.add_argument(
        "--thickness-metric",
        type=str,
        default="half",
        choices=("half", "eff"),
        help=(
            "Thickness diagnostic used in comparison plots (strip mode) and for u_L in the interface_total removal. "
            "'half' uses the geometric α=0.5 contour; "
            "'eff' uses L_eff := (∫α dA)/width (matches sharp-interface thickness when α≈1/0)."
        ),
    )
    ap.add_argument(
        "--no-clip",
        dest="clip",
        action="store_false",
        default=True,
        help="Disable post-step clipping alpha,phi,S into physical bounds (not recommended).",
    )

    # Mesh / geometry (nondimensional lengths)
    ap.add_argument("--Lx", type=float, default=0.2, help="Domain width (nondimensional).")
    ap.add_argument("--Hy", type=float, default=1.5, help="Domain height (nondimensional). Must exceed the maximum thickness.")
    ap.add_argument("--nx", type=int, default=4, help="Strip: elements in x.")
    ap.add_argument("--ny", type=int, default=240, help="Strip: elements in y.")
    ap.add_argument(
        "--y-stretch",
        type=float,
        default=0.0,
        help=(
            "Optional y-mesh stretching parameter c in y = Hy*(η + c*sin(2π η)/(2π)), η=y/Hy. "
            "Use c<0 to cluster resolution near y=0 and y=Hy (both ends)."
        ),
    )
    ap.add_argument("--nx-2d", type=int, default=32, help="2D: elements in x.")
    ap.add_argument("--ny-2d", type=int, default=64, help="2D: elements in y.")
    ap.add_argument("--q", type=int, default=8, help="Assembly quadrature order.")

    # Time stepping (days)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument(
        "--t-start",
        type=float,
        default=0.0,
        help=(
            "Start time (days). Useful to run only the sloughing window for case 4 "
            "(e.g. --t-start 5.934 --t-final 6.05 with --h0 ~ 0.68)."
        ),
    )
    ap.add_argument("--t-final", type=float, default=10.0)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument(
        "--warmup-substrate-steps",
        type=int,
        default=0,
        help=(
            "If >0 and --solve-strategy=split, run this many substrate-only warmup steps before the main time loop. "
            "Useful with --t-start>0 + thick --h0 to relax S away from the uniform initial value."
        ),
    )
    ap.add_argument(
        "--warmup-substrate-dt",
        type=float,
        default=0.01,
        help="Warmup step size (days) used with --warmup-substrate-steps.",
    )
    ap.add_argument(
        "--adaptive-dt",
        action="store_true",
        help=(
            "Enable adaptive time-step reduction on Newton failure (and optional re-increase on easy steps). "
            "Useful for the long Warner cases (2–4) where dt=0.5 can be too aggressive at early times."
        ),
    )
    ap.add_argument(
        "--dt-min",
        type=float,
        default=0.0,
        help="Minimum dt when using --adaptive-dt (0 disables the lower bound).",
    )
    ap.add_argument(
        "--dt-max",
        type=float,
        default=0.0,
        help="Maximum dt when using --adaptive-dt (0 uses the initial --dt).",
    )
    ap.add_argument(
        "--dt-after-event",
        type=float,
        default=0.01,
        help=(
            "Initial dt to use immediately after event times (days). "
            "Used when the script splits the solve into segments for case 2 (step at 6d) "
            "and case 4 (sloughing around 6d) so the post-event transient converges robustly. "
            "If <=0, reuse --dt."
        ),
    )
    ap.add_argument(
        "--dt-reduction-factor",
        type=float,
        default=0.5,
        help="dt <- factor*dt on failed step when using --adaptive-dt.",
    )
    ap.add_argument(
        "--dt-increase-factor",
        type=float,
        default=2.0,
        help="dt <- factor*dt on easy steps when using --adaptive-dt (capped by --dt-max).",
    )
    ap.add_argument(
        "--dt-iters-increase-threshold",
        type=int,
        default=25,
        help="Increase dt when Newton iterations <= this threshold (requires --adaptive-dt).",
    )
    ap.add_argument(
        "--dt-easy-steps-before-increase",
        type=int,
        default=1,
        help="Number of consecutive easy steps before increasing dt (requires --adaptive-dt).",
    )
    ap.add_argument(
        "--dt-decrease-factor-slow",
        type=float,
        default=1.0,
        help=(
            "dt <- factor*dt after slow (but converged) steps (requires --adaptive-dt). "
            "Use 1.0 to disable slow-step reductions."
        ),
    )
    ap.add_argument(
        "--dt-iters-decrease-threshold",
        type=int,
        default=40,
        help="Decrease dt after a slow (but converged) step when Newton iterations >= this threshold (requires --adaptive-dt).",
    )
    ap.add_argument(
        "--dt-slow-steps-before-decrease",
        type=int,
        default=1,
        help="Number of consecutive slow steps before decreasing dt (requires --adaptive-dt).",
    )
    ap.add_argument(
        "--dt-reject-on-slow",
        action="store_true",
        help="If set, reject a slow (but converged) step and retry at reduced dt (requires --adaptive-dt).",
    )
    ap.add_argument("--newton-tol", type=float, default=1.0e-6)
    ap.add_argument(
        "--newton-rtol",
        type=float,
        default=0.0,
        help="Optional relative Newton tolerance. Stops when |R|_inf <= max(newton_tol, newton_rtol*|R0|_inf).",
    )
    ap.add_argument("--max-it", type=int, default=40)
    ap.add_argument(
        "--no-line-search",
        dest="line_search",
        action="store_false",
        default=True,
        help="Disable line-search backtracking and always take the full Newton step (sometimes needed for saddle-point blocks).",
    )
    ap.add_argument("--ls-mode", type=str, default="dealii", choices=("armijo", "dealii"))
    ap.add_argument(
        "--pdas",
        dest="pdas",
        action="store_true",
        default=True,
        help="Use internal PDAS/semismooth Newton with box bounds for (S,alpha) (default: enabled).",
    )
    ap.add_argument(
        "--no-pdas",
        dest="pdas",
        action="store_false",
        help="Disable PDAS box constraints and use the standard Newton solver.",
    )
    ap.add_argument(
        "--no-bounds",
        dest="bounds",
        action="store_false",
        default=True,
        help="Disable box bounds on (S,alpha).",
    )
    ap.add_argument(
        "--vi-c",
        type=float,
        default=0.0,
        help="PDAS active-set scaling parameter c (>0). Use c<=0 to auto-estimate from Jacobian diagonal.",
    )
    ap.add_argument(
        "--vi-active-tol",
        type=float,
        default=0.0,
        help="Active-set tolerance used in the NCP test for PDAS (default: 0).",
    )
    ap.add_argument(
        "--vi-project-each-iter",
        dest="vi_project_each_iter",
        action="store_true",
        default=False,
        help=(
            "Project the iterate to the box bounds after each VI-Newton iteration. "
            "This can improve robustness for very stiff kinetics, but may also "
            "slow down or stall convergence near the tolerance. Default: off."
        ),
    )

    # Nondimensionalization: x_phys = L_ref * x_nondim.
    ap.add_argument("--L-ref-m", type=float, default=1.0e-3, help="Length scale for output conversion (meters).")

    # Initial conditions (nondimensional thickness)
    ap.add_argument("--h0", type=float, default=0.005, help="Initial biofilm thickness h0 (nondimensional).")
    ap.add_argument("--eps0", type=float, default=0.01, help="Initial interface thickness epsilon for alpha0 (nondimensional).")
    ap.add_argument("--phi-b", type=float, default=0.3, help="Initial biofilm porosity (phi inside biofilm where alpha≈1).")
    ap.add_argument("--S-high", type=float, default=3.0, help="Bulk/surface COD concentration S_L1 before step (g/m^3).")
    ap.add_argument(
        "--case2-autotroph",
        action="store_true",
        help=(
            "Case 2 only: add an effective COD-independent expansion term after the COD step (t>=6d) "
            "to mimic Warner's multispecies regrowth. This modifies only the mixture volume constraint "
            "source s_v (no COD consumption is added)."
        ),
    )
    ap.add_argument("--case2-autotroph-start", type=float, default=6.0, help="Start time for the effective autotroph expansion term (days).")
    ap.add_argument("--case2-autotroph-mu0", type=float, default=0.173, help="Expansion rate at t=start (1/d).")
    ap.add_argument("--case2-autotroph-mu1", type=float, default=0.249, help="Asymptotic expansion rate as t→∞ (1/d).")
    ap.add_argument("--case2-autotroph-tau", type=float, default=3.0, help="Ramp timescale tau (days) for the exponential transition mu0→mu1.")
    ap.add_argument(
        "--mu-extra-const",
        type=float,
        default=0.0,
        help=(
            "Optional COD-independent extra volumetric expansion rate (1/d) added to the mixture volume constraint source s_v. "
            "This does NOT add COD consumption. "
            "Use this only as part of the reduced mapping when the single-substrate model must emulate additional growth/inert "
            "accumulation from unmodeled substrates/species (Warner cases are multispecies/multisubstrate). "
            "Case 2: this value is added on top of --case2-autotroph (if enabled)."
        ),
    )
    ap.add_argument(
        "--case5-Q",
        type=float,
        default=0.5,
        help="Case 5: reactor feed flow rate Q (m^3/d).",
    )
    ap.add_argument(
        "--case5-VR",
        type=float,
        default=0.01,
        help="Case 5: reactor volume VR (m^3).",
    )
    ap.add_argument(
        "--case5-AL",
        type=float,
        default=1.0,
        help="Case 5: biofilm area AL (m^2).",
    )
    ap.add_argument(
        "--case5-LL",
        type=float,
        default=2.0e-5,
        help="Case 5: laminar sublayer thickness LL (m). Used only in VL=VR-AL(L+LL).",
    )

    # Model parameters (single substrate, COD-like)
    ap.add_argument("--D-S-phys", type=float, default=83e-6, help="Physical diffusion D (m^2/d) used to derive nondim D_S.")
    ap.add_argument(
        "--substrate-reaction-scheme",
        type=str,
        default="theta",
        choices=("theta", "implicit", "explicit"),
        help=(
            "Time discretization for the stiff substrate reaction term R_S in the substrate equation. "
            "'theta' uses the global --theta (e.g. CN when theta=0.5). "
            "'implicit' treats R_S fully implicitly (IMEX, L-stable for stiff decay). "
            "'explicit' treats R_S explicitly (not recommended)."
        ),
    )
    ap.add_argument(
        "--substrate-diffusion-scheme",
        type=str,
        default="theta",
        choices=("theta", "implicit", "explicit"),
        help=(
            "Time discretization for the substrate diffusion term -div(D grad S). "
            "'theta' uses the global --theta (e.g. CN when theta=0.5). "
            "'implicit' treats diffusion fully implicitly (L-stable damping of high-frequency modes). "
            "'explicit' treats diffusion explicitly (not recommended)."
        ),
    )
    ap.add_argument("--D-alpha", type=float, default=0.0, help="Nondimensional alpha diffusion coefficient (set 0 when using Allen–Cahn regularization).")
    ap.add_argument(
        "--alpha-advection-form",
        type=str,
        default="advective",
        choices=("advective", "conservative"),
        help="Alpha transport by vS: 'advective' uses vS·grad(alpha) (indicator); 'conservative' uses div(alpha*vS).",
    )
    ap.add_argument("--D-phi", type=float, default=3.0e-3, help="Nondimensional phi diffusion coefficient.")
    ap.add_argument("--gamma-phi", type=float, default=10.0, help="Penalty driving phi->1 in fluid (alpha≈0).")
    ap.add_argument("--alpha-cahn-M", type=float, default=0.1, help="Allen–Cahn mobility for alpha (0 disables).")
    ap.add_argument("--alpha-cahn-gamma", type=float, default=1.0e-4, help="Allen–Cahn surface-energy coefficient for alpha (0 disables).")
    ap.add_argument(
        "--alpha-cahn-eps",
        type=float,
        default=-1.0,
        help="Allen–Cahn interface thickness epsilon (nondimensional). Default: use --eps0.",
    )
    ap.add_argument(
        "--alpha-cahn-mobility",
        type=str,
        default="constant",
        choices=("constant", "degenerate"),
        help="Mobility for Allen–Cahn terms: 'constant' or 'degenerate' (M(alpha)=M0*alpha*(1-alpha)).",
    )
    ap.add_argument(
        "--alpha-cahn-conservative",
        dest="alpha_cahn_conservative",
        action="store_true",
        default=True,
        help="Use conservative Allen–Cahn for alpha via a global lambda_alpha (default: enabled).",
    )
    ap.add_argument(
        "--no-alpha-cahn-conservative",
        dest="alpha_cahn_conservative",
        action="store_false",
        help="Disable conservative Allen–Cahn (not recommended when D_alpha=0).",
    )
    ap.add_argument(
        "--alpha-cahn-conservative-mode",
        type=str,
        default="eliminate",
        choices=("eliminate", "unknown"),
        help="How to handle lambda_alpha: 'eliminate' projects lambda each assembly; 'unknown' solves it as a global unknown.",
    )
    ap.add_argument("--gamma-u", type=float, default=5.0, help="u-extension penalty factor in the free-fluid region.")
    ap.add_argument("--u-extension", type=str, default="l2", choices=("l2", "grad"), help="u-extension mode (see model.tex).")
    ap.add_argument("--gamma-u-pin", type=float, default=1.0e-4, help="Tiny L2 pin used with --u-extension grad.")
    ap.add_argument("--rho-f", type=float, default=1.0, help="Mixture/fluid density scale (nondimensional).")
    ap.add_argument("--mu-f", type=float, default=1.0e-2, help="Fluid viscosity scale (nondimensional).")
    ap.add_argument("--kappa-inv", type=float, default=10.0, help="Inverse permeability scale (nondimensional).")
    ap.add_argument("--mu-s", type=float, default=0.0, help="Skeleton shear modulus scale (nondimensional).")
    ap.add_argument("--lambda-s", type=float, default=0.0, help="Skeleton bulk modulus scale (nondimensional).")
    ap.add_argument(
        "--solid-visco-eta",
        type=float,
        default=0.0,
        help="Kelvin–Voigt viscosity coefficient for the skeleton (nondimensional).",
    )
    ap.add_argument("--mu-max", type=float, default=4.8, help="Monod max rate (1/d).")
    ap.add_argument("--K-S", type=float, default=5.0, help="Monod saturation (g/m^3).")
    ap.add_argument(
        "--k-g-mode",
        type=str,
        default="constant",
        choices=("constant", "warner"),
        help=(
            "How to apply --k-g in the alpha growth term G(S,phi). "
            "'constant' uses k_g as-is. "
            "'warner' scales k_g each step as k_g_eff = k_g * (2*L_eff/eps_alpha) "
            "so the diffuse-interface growth reproduces the sharp-interface law dL/dt≈mu_bar*L in 1D."
        ),
    )
    ap.add_argument("--k-g", type=float, default=0.0, help="Interface growth prefactor in G(S,phi) (set 0 to rely on volumetric expansion).")
    ap.add_argument("--k-d", type=float, default=0.2, help="Decay rate (1/d) in Pi_b.")
    ap.add_argument("--Y", type=float, default=0.4, help="Yield.")
    ap.add_argument("--rho-s-star", type=float, default=5000.0, help="Intrinsic solid density (g/m^3).")
    ap.add_argument(
        "--rho-s-effective",
        type=float,
        default=0.0,
        help=(
            "If >0, override --rho-s-star so that the effective biomass density inside the biofilm "
            "rho_eff := (1-phi_b)*rho_s_star matches this value (g/m^3). "
            "Example: with phi_b=0.3 and rho_s_effective=5000, we set rho_s_star=7142.857."
        ),
    )
    ap.add_argument(
        "--p-fluid-penalty",
        type=float,
        default=1.0e-2,
        help=(
            "Pressure regularization in the fluid region when v is frozen (mechanics='skeleton'): "
            "adds γ_p (1-α)^16 p q to remove decoupled p-DOFs. Set 0 to disable."
        ),
    )

    # Detachment/sloughing mapping
    ap.add_argument(
        "--eps-det",
        type=float,
        default=-1.0,
        help="Effective interface thickness used in V≈4 eps D_det (nondimensional). Default: use --eps0.",
    )
    ap.add_argument("--lambda-shear", type=float, default=750.0, help="Warner shear coefficient lambda (m^-1 d^-1) for case 3 mapping.")
    ap.add_argument(
        "--slough-mode",
        type=str,
        default="integrated",
        choices=("integrated", "exact", "shift", "shift_window"),
        help=(
            "Case 4 sloughing: "
            "'shift' applies an instantaneous thickness drop at t=5.994 (end of Warner's window); "
            "'shift_window' applies the same drop gradually over [5.984, 5.994] by repeated small shifts; "
            "'integrated' applies a coarse drop around t=6; "
            "'exact' uses Warner's (5.984,5.994)d detachment window."
        ),
    )
    ap.add_argument("--slough-drop-um", type=float, default=500.0, help="Case 4 integrated sloughing thickness drop (microns).")

    # 2D initial surface perturbation
    ap.add_argument("--h0-amp", type=float, default=0.0, help="2D: relative amplitude for wavy initial thickness (0.2 => ±20 percent).")
    ap.add_argument(
        "--surface-gamma",
        type=float,
        default=0.0,
        help=(
            "Optional interface Dirichlet-penalty for the substrate: adds "
            "γ_s |∇α| (S - S_bulk) to the substrate residual to enforce S≈S_bulk at the "
            "diffuse interface (Warner cases 1–4 assume fixed surface concentration). "
            "Units: 1/d. Use 0 to disable."
        ),
    )
    ap.add_argument(
        "--surface-jacobian",
        type=str,
        default="frozen",
        choices=("full", "frozen"),
        help=(
            "Linearization mode for the interface penalty term added by --surface-gamma. "
            "'full' includes α-dependence of |∇α| in the Jacobian; "
            "'frozen' treats |∇α| as fixed (Picard), recommended for --solve-strategy split."
        ),
    )

    # NOTE: argparse uses old-style '%' string formatting internally. Avoid '%' in help strings.
    args = ap.parse_args()

    # Keep the benchmark logs readable: suppress very verbose assembly INFO logs.
    logging.getLogger("pycutfem.ufl.compilers").setLevel(logging.WARNING)

    t_start_global = float(getattr(args, "t_start", 0.0) or 0.0)
    if not math.isfinite(t_start_global) or t_start_global < 0.0:
        raise ValueError(f"--t-start must be a finite value >= 0; got {args.t_start!r}.")
    if float(t_start_global) >= float(args.t_final) - _TIME_TOL_DAYS:
        raise ValueError(f"--t-start must be < --t-final (got {t_start_global:g} >= {float(args.t_final):g}).")

    if float(getattr(args, "rho_s_effective", 0.0) or 0.0) > 0.0:
        one_m_phi_b = 1.0 - float(args.phi_b)
        if one_m_phi_b <= 0.0:
            raise ValueError(f"--rho-s-effective requires phi_b<1; got phi_b={float(args.phi_b):.6g}.")
        args.rho_s_star = float(args.rho_s_effective) / one_m_phi_b
        print(
            "[preset] rho_s_star overridden from --rho-s-effective: "
            f"rho_s_star={float(args.rho_s_star):.6g} so (1-phi_b)*rho_s_star={float(args.rho_s_effective):.6g}",
            flush=True,
        )

    # In the Warner ζ-model there is no elastic skeleton response. For the one-domain
    # benchmark, setting (mu_s,lambda_s)=(0,0) makes vS determined by (constraint + drag),
    # and the kinematics block for u becomes unnecessary. Freezing u significantly
    # improves Newton/PDAS robustness while keeping the intended kinematic thickness
    # evolution driven by vS.
    if (
        str(args.mechanics).strip().lower() in {"skeleton", "full"}
        and float(args.mu_s) == 0.0
        and float(args.lambda_s) == 0.0
        and not bool(args.freeze_u)
    ):
        args.freeze_u = True
        print(
            f"[preset] mechanics={args.mechanics} with mu_s=lambda_s=0 -> enabling --freeze-u for robustness.",
            flush=True,
        )

    mech_key = str(args.mechanics).strip().lower()
    sv_key = str(getattr(args, "s_v_mode", "auto")).strip().lower()
    if sv_key == "auto":
        args.s_v_mode = "mu" if mech_key == "full" else "pi"
        print(f"[preset] s_v_mode=auto -> using --s-v-mode {args.s_v_mode} for mechanics={args.mechanics}.", flush=True)
    elif sv_key == "pi" and mech_key == "full":
        print(
            "[warn] mechanics=full with s_v_mode=pi double-counts (1-phi) in the volumetric source and typically underpredicts growth; "
            "use --s-v-mode mu for full mechanics.",
            flush=True,
        )
    elif sv_key == "mu" and mech_key in {"skeleton", "frozen"}:
        print(
            "[warn] mechanics!=full with s_v_mode=mu can overdrive vS because the constraint involves div(B vS) rather than div(vS); "
            "use --s-v-mode pi (or set --s-v-mode auto).",
            flush=True,
        )

    if float(getattr(args, "k_g", 0.0) or 0.0) == 0.0 and str(getattr(args, "alpha_advection_form", "advective")).strip().lower() == "conservative":
        print(
            "[warn] alpha_advection_form=conservative with k_g=0 conserves alpha under expansion; "
            "thickness often does not grow. Use --alpha-advection-form advective (indicator advection) "
            "or set --k-g>0 to include a growth source in the alpha equation.",
            flush=True,
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paper_figdir = Path(args.paper_figdir).expanduser().resolve() if args.paper_figdir else None
    if paper_figdir is not None:
        paper_figdir.mkdir(parents=True, exist_ok=True)

    backend = str(args.backend)
    qdeg = int(args.q)
    theta = float(args.theta)
    L_ref_m = float(args.L_ref_m)

    # Crank–Nicolson (θ=0.5) is not L-stable. For the Warner parameter ranges the
    # substrate sink can be extremely stiff once scaled by ρ_s^*/Y, so using θ
    # for the substrate diffusion/reaction terms may cause oscillations and poor
    # VI-Newton convergence unless Δt is extremely small. Recommend the IMEX
    # options by default when θ<1.
    if theta < 1.0:
        r_scheme = str(getattr(args, "substrate_reaction_scheme", "theta")).strip().lower()
        d_scheme = str(getattr(args, "substrate_diffusion_scheme", "theta")).strip().lower()
        if r_scheme == "theta" or d_scheme == "theta":
            print(
                "[warn] theta<1 with substrate_{reaction,diffusion}_scheme='theta' can be unstable for stiff Monod kinetics. "
                "Try --substrate-reaction-scheme implicit --substrate-diffusion-scheme implicit.",
                flush=True,
            )

    warner_outdir = Path("examples/biofilms/results/warner1986").resolve()
    cases = [1, 2, 3, 4, 5] if args.case == "all" else [int(args.case)]

    # Nondimensional substrate diffusion (time in days, length in L_ref meters)
    D_S_nondim = float(args.D_S_phys) / (float(L_ref_m) * float(L_ref_m))

    # For consistency with Warner, always compare to the cpp-run reference.
    ref_by_case = {cid: _load_warner_cod_reference(case_id=cid, warner_outdir=warner_outdir) for cid in cases}

    rows_by_mode_case: dict[tuple[str, int], list[dict[str, float]]] = {}

    def _write_rows_csv(*, rows: list[dict[str, float]], out_csv: Path) -> None:
        if not rows:
            return
        cols = [
            "t_days",
            "L_eff_um",
            "L_half_um",
            "L_thickness_um",
            "removal",
            "removal_top_flux",
            "removal_bulk_exchange",
            "removal_surface_exchange",
            "removal_reservoir_exchange",
            "removal_consumption",
            "removal_interface_diff_grad",
            "removal_interface_diff_int",
            "removal_interface_diff",
            "removal_interface_total",
            "uL_um_per_d",
            "D_det_prev",
            "k_g_eff",
            "S_surf",
            "S_top",
            "mu_extra",
        ]
        arr = np.column_stack([[r[c] for r in rows] for c in cols])
        tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
        np.savetxt(tmp, arr, delimiter=",", header=",".join(cols), comments="")
        os.replace(tmp, out_csv)

    def _run_mode(mode: str, *, cid: int, Lx: float, Hy: float, nx: int, ny: int, h0_fn) -> Path:
        D_det_prev = Constant(0.0)
        t_start = float(getattr(args, "t_start", 0.0) or 0.0)
        if int(cid) == 5 and abs(float(t_start)) > _TIME_TOL_DAYS:
            raise NotImplementedError("Case 5 with --t-start>0 is not supported (bulk ODE restart not implemented).")

        S_bulk_value = float(args.S_high) if int(cid) == 5 else float(_S_bulk_case(cid, float(t_start), S_high=float(args.S_high)))
        S_bulk_c = Constant(float(S_bulk_value))
        gamma_bulk_c = Constant(float(args.bulk_gamma))
        k_g_base = float(args.k_g)
        k_g_c = Constant(float(k_g_base))
        k_g_mode = str(getattr(args, "k_g_mode", "constant")).strip().lower()
        if k_g_mode in {"const", "constant"}:
            k_g_mode = "constant"
        elif k_g_mode in {"warner", "thickness", "thickness_scaled", "scaled"}:
            k_g_mode = "warner"
        else:
            raise ValueError(f"Unknown --k-g-mode={args.k_g_mode!r}. Use 'constant' or 'warner'.")
        eps_ac = float(args.alpha_cahn_eps)
        if eps_ac <= 0.0:
            eps_ac = float(args.eps0)
        eps_det_eff = float(args.eps_det) if float(args.eps_det) > 0.0 else float(args.eps0)

        prob = _build_one_domain_problem(
            Lx=float(Lx),
            Hy=float(Hy),
            nx=int(nx),
            ny=int(ny),
            y_stretch=float(getattr(args, "y_stretch", 0.0) or 0.0),
            qdeg=int(qdeg),
            dt_days=float(args.dt),
            theta=float(theta),
            backend=backend,
            mechanics_mode=str(args.mechanics),
            freeze_alpha=bool(getattr(args, "freeze_alpha", False)),
            freeze_u=bool(getattr(args, "freeze_u", False)),
            freeze_vSx=bool(getattr(args, "freeze_vSx", False)),
            freeze_phi=bool(args.freeze_phi),
            s_v_mode=str(args.s_v_mode),
            s_v_lagged=bool(getattr(args, "s_v_lagged", False)),
            s_v_jacobian=str(getattr(args, "s_v_jacobian", "full")),
            gamma_u=float(args.gamma_u),
            u_extension_mode=str(args.u_extension),
            gamma_u_pin=float(args.gamma_u_pin),
            alpha_cahn_M=float(args.alpha_cahn_M),
            alpha_cahn_gamma=float(args.alpha_cahn_gamma),
            alpha_cahn_eps=float(eps_ac),
            alpha_cahn_mobility=str(args.alpha_cahn_mobility),
            alpha_cahn_conservative=bool(args.alpha_cahn_conservative),
            alpha_cahn_conservative_mode=str(args.alpha_cahn_conservative_mode),
            h0=float(args.h0),
            eps0=float(args.eps0),
            phi_b=float(args.phi_b),
            S0=float(args.S_high),
            init_vS_profile=str(getattr(args, "init_vS", "growth")),
            D_S=float(D_S_nondim),
            substrate_reaction_scheme=str(getattr(args, "substrate_reaction_scheme", "theta")),
            substrate_diffusion_scheme=str(getattr(args, "substrate_diffusion_scheme", "theta")),
            D_alpha=float(args.D_alpha),
            alpha_advection_form=str(getattr(args, "alpha_advection_form", "advective")),
            D_phi=float(args.D_phi),
            gamma_phi=float(args.gamma_phi),
            rho_f=float(args.rho_f),
            mu_f=float(args.mu_f),
            kappa_inv=float(args.kappa_inv),
            mu_s=float(args.mu_s),
            lambda_s=float(args.lambda_s),
            solid_visco_eta=float(getattr(args, "solid_visco_eta", 0.0) or 0.0),
            mu_max=float(args.mu_max),
            K_S=float(args.K_S),
            k_g=k_g_c,
            k_d=float(args.k_d),
            Y=float(args.Y),
            rho_s_star=float(args.rho_s_star),
            D_det_prev=D_det_prev,
        )

        dh: DofHandler = prob["dh"]
        forms = prob["forms"]
        bcs = list(prob["bcs"])
        bcs_homog = list(prob["bcs_homog"])
        funcs = list(prob["functions"])
        prev_funcs = list(prob["prev_functions"])
        fields = dict(prob["fields"])
        S_k: Function = prob["S_k"]
        alpha_k: Function = prob["alpha_k"]
        S_n: Function = prob["S_n"]
        dt_c: Constant = prob["dt_c"]
        mu_extra_c: Constant = prob["mu_extra_c"]
        alpha_n: Function = prob["alpha_n"]

        # Initialize COD-independent volumetric expansion (if enabled) at t_start
        # so the first step uses the correct value when --t-start>0.
        mu_extra_base = float(getattr(args, "mu_extra_const", 0.0) or 0.0)
        if int(cid) == 2:
            mu_extra_c.value = mu_extra_base + _case2_mu_extra(
                t_days=float(t_start),
                enabled=bool(getattr(args, "case2_autotroph", False)),
                start_days=float(getattr(args, "case2_autotroph_start", 6.0)),
                mu0=float(getattr(args, "case2_autotroph_mu0", 0.0)),
                mu1=float(getattr(args, "case2_autotroph_mu1", 0.0)),
                tau_days=float(getattr(args, "case2_autotroph_tau", 1.0)),
            )
        else:
            mu_extra_c.value = mu_extra_base

        # Override initial alpha for 2D mode (strip stays uniform).
        if callable(h0_fn):
            alpha0 = lambda x, y: _alpha_step_eval(x, y, h=h0_fn(np.asarray(x, dtype=float)), eps=float(args.eps0))
            alpha_prev: Function = prob["alpha_n"]
            alpha_prev.set_values_from_function(alpha0)
            alpha_k.nodal_values[:] = alpha_prev.nodal_values[:]
            phi_prev: Function = prob["phi_n"]
            phi_prev.set_values_from_function(lambda x, y: float(args.phi_b))
            prob["phi_k"].nodal_values[:] = phi_prev.nodal_values[:]

        # Substrate BC on the top boundary (time-dependent for case 2).
        def _S_top_bc(x, y, t):
            if int(cid) == 5:
                return float(S_bulk_value)
            return _S_bulk_case(cid, float(t), S_high=float(args.S_high))

        bcs.append(BoundaryCondition("S", "dirichlet", "top", _as_float_time(_S_top_bc)))
        bcs_homog.append(BoundaryCondition("S", "dirichlet", "top", (lambda x, y: 0.0)))

        n = FacetNormal()
        width = float(Lx)
        dx_q = dx(metadata={"q": int(qdeg)})

        # Optional "well-mixed bulk" relaxation for substrate in the fluid region.
        #
        # Motivation: Warner cases 1--4 specify substrate concentrations at the
        # film-water interface (i.e. effectively Dirichlet at the interface with
        # no external mass-transfer resistance). In a fixed-domain one-domain
        # simulation with an explicit fluid region above the film, imposing
        # Dirichlet only on the top boundary introduces an artificial diffusion
        # sublayer whose thickness grows with (Hy - L). The relaxation term below
        # emulates a well-mixed bulk by driving S -> S_bulk(t) wherever alpha≈0.
        bulk_mode = str(args.bulk_mode).strip().lower()
        if bulk_mode not in {"top", "well_mixed"}:
            raise ValueError(f"Unknown bulk_mode={args.bulk_mode!r}. Use 'top' or 'well_mixed'.")
        base_residual_form = forms.residual_form
        base_jacobian_form = forms.jacobian_form

        # If the mixture velocity v is *not* solved (mechanics='skeleton'), then p-DOFs
        # in the pure-fluid region (α≈0) can become weakly/fully decoupled. This leads
        # to singular or ill-conditioned Newton systems. A small pressure penalty
        # localized to the fluid region fixes the algebraic nullspace without affecting
        # the biofilm region (α≈1).
        mech_key = str(args.mechanics).strip().lower()
        gamma_p = float(getattr(args, "p_fluid_penalty", 0.0) or 0.0)
        if mech_key == "skeleton" and gamma_p != 0.0:
            gamma_p_c = Constant(float(gamma_p))
            q_p = TestFunction("p", dof_handler=dh)
            dp = TrialFunction("p", dof_handler=dh)
            dalpha = TrialFunction("alpha", dof_handler=dh)
            one_m_alpha_k = _one_minus(alpha_k)
            p_k = fields["p"]
            w2 = one_m_alpha_k * one_m_alpha_k
            w4 = w2 * w2
            w8 = w4 * w4
            w16 = w8 * w8
            dw16 = (-Constant(16.0) * w8 * w4 * w2 * one_m_alpha_k) * dalpha  # -16(1-α)^15 δα
            base_residual_form = base_residual_form + gamma_p_c * w16 * p_k * q_p * dx_q
            base_jacobian_form = base_jacobian_form + gamma_p_c * (w16 * dp + dw16 * p_k) * q_p * dx_q
        bulk_residual = None
        bulk_jacobian = None
        w_bulk = Constant(0.0)
        if bulk_mode == "well_mixed" and float(args.bulk_gamma) != 0.0:
            S_test = TestFunction("S", dof_handler=dh)
            dS_trial = TrialFunction("S", dof_handler=dh)
            dalpha = TrialFunction("alpha", dof_handler=dh)
            one_m_alpha_k = _one_minus(alpha_k)
            m_bulk = int(getattr(args, "bulk_alpha_power", 16) or 0)
            w_bulk = _pow_int(one_m_alpha_k, m_bulk)
            if m_bulk <= 0:
                dw_bulk = Constant(0.0) * dalpha
            else:
                dw_bulk = (-Constant(float(m_bulk)) * _pow_int(one_m_alpha_k, m_bulk - 1)) * dalpha
            bulk_residual = gamma_bulk_c * w_bulk * (S_k - S_bulk_c) * S_test * dx_q
            bulk_jac_mode = str(getattr(args, "bulk_jacobian", "full")).strip().lower()
            if bulk_jac_mode in {"full"}:
                bulk_jacobian = gamma_bulk_c * (w_bulk * dS_trial + (S_k - S_bulk_c) * dw_bulk) * S_test * dx_q
            elif bulk_jac_mode in {"frozen", "picard"}:
                bulk_jacobian = gamma_bulk_c * (w_bulk * dS_trial) * S_test * dx_q
            else:
                raise ValueError(f"Unknown --bulk-jacobian={args.bulk_jacobian!r}. Use 'full' or 'frozen'.")

        # Optional interface Dirichlet-penalty (surface BC at α=0.5):
        #   ∫ γ_s |∇α| (S - S_bulk) w dx  ≈  ∫_Γ γ_s (S - S_bulk) w dS.
        #
        # This addresses the main mismatch for Warner cases 1–4: the FD reference uses
        # a fixed surface concentration S(t,L)=S_L(t), while a fixed-domain diffuse
        # interface has a finite-thickness transition region where reaction can
        # reduce S at the α=0.5 contour. The penalty enforces the Dirichlet value
        # at the interface without injecting substrate into the bulk volume.
        surface_residual = None
        surface_jacobian = None
        gamma_surf_c = Constant(0.0)
        w_surf = Constant(0.0)
        surface_gamma = float(getattr(args, "surface_gamma", 0.0) or 0.0)
        if surface_gamma != 0.0:
            gamma_surf_c = Constant(float(surface_gamma))
            S_test = TestFunction("S", dof_handler=dh)
            dS_trial = TrialFunction("S", dof_handler=dh)
            dalpha = TrialFunction("alpha", dof_handler=dh)
            g2 = inner(grad(alpha_k), grad(alpha_k))
            w_surf = (g2 + Constant(1.0e-16)) ** Constant(0.5)
            surface_residual = gamma_surf_c * w_surf * (S_k - S_bulk_c) * S_test * dx_q

            surf_jac_mode = str(getattr(args, "surface_jacobian", "frozen")).strip().lower()
            if surf_jac_mode in {"full"}:
                # δ|∇α| = (∇α·∇δα)/|∇α| for |∇α| = sqrt(∇α·∇α + η).
                dw_surf = (inner(grad(alpha_k), grad(dalpha)) / w_surf) if surface_gamma != 0.0 else Constant(0.0) * dalpha
                surface_jacobian = gamma_surf_c * (w_surf * dS_trial + (S_k - S_bulk_c) * dw_surf) * S_test * dx_q
            elif surf_jac_mode in {"frozen", "picard"}:
                surface_jacobian = gamma_surf_c * (w_surf * dS_trial) * S_test * dx_q
            else:
                raise ValueError(f"Unknown --surface-jacobian={args.surface_jacobian!r}. Use 'full' or 'frozen'.")

        # Net substrate transfer from the (imposed) bulk into the computational domain.
        # Positive when the bulk supplies substrate (S < S_bulk), negative when substrate is released.
        I_bulk_exchange = (-gamma_bulk_c * w_bulk * (S_k - S_bulk_c)) * dx_q
        # If the interface penalty is enabled, it also represents exchange with the
        # imposed bulk reservoir (localized to the diffuse interface).
        I_surface_exchange = (-gamma_surf_c * w_surf * (S_k - S_bulk_c)) * dx_q

        rows: list[dict[str, float]] = []
        run_tag = _sanitize_run_tag(getattr(args, "run_tag", ""))
        tag_suffix = f"_{run_tag}" if run_tag else ""
        out_csv = outdir / f"one_domain_{mode}_case{cid}_backend={backend}{tag_suffix}_timeseries.csv"

        dS_top = dS(tag="top", metadata={"q": int(qdeg)})
        D_S_c = Constant(float(D_S_nondim))
        I_alpha = alpha_k * dx_q
        I_alpha01 = (alpha_k * _one_minus(alpha_k)) * dx_q
        I_flux_top = (D_S_c * dot(grad(S_k), n)) * dS_top
        I_flux_interface = (-D_S_c * dot(grad(S_k), grad(alpha_k))) * dx_q
        # Total consumption (g/m^3/d) integrated over volume, divided by width to get flux-like removal.
        mu_max_c = Constant(float(args.mu_max))
        K_S_c = Constant(float(args.K_S))
        k_d_c = Constant(float(args.k_d))
        Y_c = Constant(float(args.Y))
        rho_s_star_c = Constant(float(args.rho_s_star))
        monod_k = mu_max_c * (S_k / (S_k + K_S_c))
        Pi_over_rho_s_k = (monod_k - k_d_c) * _one_minus(funcs[4]) * alpha_k  # (1-phi)*alpha
        RS_k = (rho_s_star_c / Y_c) * Pi_over_rho_s_k
        I_consumption = RS_k * dx_q

        # For the Warner-mapping growth option, update k_g to mimic the sharp-interface law
        # dL/dt ≈ mu_bar * L when using a diffuse-interface source G alpha(1-alpha).
        #
        # For a general diffuse profile, with vS=0 and no detachment,
        #   d/dt ∫α dx = ∫ G α(1-α) dx.
        # In the strip, L_eff := (∫α dx)/width and I01 := (∫α(1-α) dx)/width, so:
        #   dL_eff/dt = G_avg * I01.
        # Choosing G_avg := mu_bar * (L_eff / I01) yields the desired growth
        #   dL_eff/dt = mu_bar * L_eff.
        if k_g_mode == "warner" and k_g_base != 0.0:
            try:
                A_alpha0 = assemble_scalar(dh, I_alpha, backend=backend, quad_order=int(qdeg))
                L0_nondim = float(A_alpha0) / max(width, 1.0e-16)
                A_01_0 = assemble_scalar(dh, I_alpha01, backend=backend, quad_order=int(qdeg))
                I01_0 = float(A_01_0) / max(width, 1.0e-16)
                k_g_c.value = float(k_g_base) * (float(L0_nondim) / max(float(I01_0), 1.0e-16))
            except Exception:
                k_g_c.value = float(k_g_base)

        L_thickness_prev_nondim: float | None = None
        slough_mode_key = str(getattr(args, "slough_mode", "integrated")).strip().lower()
        slough_shift_applied = False
        slough_shift_window_applied_nondim = 0.0
        slough_drop_nondim = float(args.slough_drop_um) * 1.0e-6 / float(L_ref_m)

        def _record_step(*, t_k: float, _funcs):
            nonlocal S_bulk_value
            nonlocal L_thickness_prev_nondim
            nonlocal slough_shift_applied
            # Update bulk substrate value for the current step (case 2 step change).
            if int(cid) != 5:
                S_bulk_value = float(_S_bulk_case(cid, float(t_k), S_high=float(args.S_high)))
                S_bulk_c.value = float(S_bulk_value)

            # Case 4 instantaneous sloughing ("shift").
            if int(cid) == 4 and slough_mode_key == "shift":
                slough_shift_applied = _apply_case4_shift_slough_strip(
                    t_days=float(t_k),
                    applied=bool(slough_shift_applied),
                    dh=dh,
                    alpha=alpha_k,
                    S=S_k,
                    drop_nondim=float(slough_drop_nondim),
                    eps_nondim=float(args.eps0),
                    S_bulk_value=float(S_bulk_value),
                )

            if bool(args.clip):
                # Post-step VI clip (bounds preservation). See model.tex discussion.
                alpha_k.nodal_values[:] = np.clip(alpha_k.nodal_values, 0.0, 1.0)
                _funcs[4].nodal_values[:] = np.clip(_funcs[4].nodal_values, 0.0, 1.0)  # phi
                S_k.nodal_values[:] = np.maximum(S_k.nodal_values, 0.0)

            k_g_used = float(getattr(k_g_c, "value", k_g_base))

            A_alpha = assemble_scalar(dh, I_alpha, backend=backend, quad_order=int(qdeg))
            L_eff_nondim = float(A_alpha) / max(width, 1.0e-16)
            L_eff_um = 1.0e6 * float(L_ref_m) * L_eff_nondim
            S_surf = float("nan")
            dS_dy_surf = float("nan")
            if mode == "strip":
                L_half_nondim, S_surf, dS_dy_surf = _strip_interface_metrics(dh=dh, alpha=alpha_k, S=S_k)
            else:
                L_half_nondim = _strip_thickness_alpha_half(dh=dh, alpha=alpha_k)
            L_half_um = 1.0e6 * float(L_ref_m) * float(L_half_nondim)
            thickness_key = str(getattr(args, "thickness_metric", "half")).strip().lower()
            if thickness_key not in {"half", "eff"}:
                raise ValueError(f"Unknown --thickness-metric={args.thickness_metric!r}. Use 'half' or 'eff'.")
            L_thickness_nondim = float(L_half_nondim) if thickness_key == "half" else float(L_eff_nondim)
            L_thickness_um = 1.0e6 * float(L_ref_m) * float(L_thickness_nondim)

            # Removal flux metrics (positive for uptake/removal).
            F_top = assemble_scalar(dh, I_flux_top, backend=backend, quad_order=int(qdeg))
            removal_top_flux = float(L_ref_m) * (float(F_top) / max(width, 1.0e-16))

            F_bulk = assemble_scalar(dh, I_bulk_exchange, backend=backend, quad_order=int(qdeg))
            removal_bulk_exchange = float(L_ref_m) * (float(F_bulk) / max(width, 1.0e-16))
            F_surf = assemble_scalar(dh, I_surface_exchange, backend=backend, quad_order=int(qdeg))
            removal_surface_exchange = float(L_ref_m) * (float(F_surf) / max(width, 1.0e-16))
            removal_reservoir_exchange = float(removal_top_flux + removal_bulk_exchange + removal_surface_exchange)

            C_tot = assemble_scalar(dh, I_consumption, backend=backend, quad_order=int(qdeg))
            removal_consumption = float(L_ref_m) * (float(C_tot) / max(width, 1.0e-16))

            # Interface diffusive flux (-j_diff) at α=0.5. For strip runs we estimate it
            # from the x-averaged 1D profile; otherwise fall back to a diffuse-interface
            # proxy (may be noisy on coarse meshes).
            removal_interface_diff_grad = float("nan")
            if mode == "strip" and np.isfinite(float(dS_dy_surf)):
                removal_interface_diff_grad = float(L_ref_m) * float(D_S_nondim) * float(dS_dy_surf)

            F_int = assemble_scalar(dh, I_flux_interface, backend=backend, quad_order=int(qdeg))
            removal_interface_diff_int = float(L_ref_m) * (float(F_int) / max(width, 1.0e-16))

            removal_interface_diff = removal_interface_diff_grad if np.isfinite(removal_interface_diff_grad) else removal_interface_diff_int

            # Warner's total interface flux is j_L = j_diff - u_L S_L.
            # Their plotted "removal" is -j_L = (-j_diff) + u_L S_L.
            dt_step = float(getattr(dt_c, "value", float(args.dt)))
            if dt_step <= 0.0:
                dt_step = float(args.dt)
            uL_m_per_d = 0.0
            if L_thickness_prev_nondim is not None and dt_step > 0.0:
                uL_m_per_d = (float(L_thickness_nondim) - float(L_thickness_prev_nondim)) * float(L_ref_m) / float(dt_step)
            L_thickness_prev_nondim = float(L_thickness_nondim)
            S_L = float(S_surf) if np.isfinite(float(S_surf)) else float(S_bulk_value)
            removal_interface_total = float(removal_interface_diff + uL_m_per_d * S_L)

            removal_key = str(args.removal_metric).strip().lower()
            if removal_key in {"consumption"}:
                removal = removal_consumption
            elif removal_key in {"top_flux", "top"}:
                removal = removal_top_flux
            elif removal_key in {"bulk_exchange", "bulk"}:
                removal = removal_bulk_exchange
            elif removal_key in {"reservoir_exchange", "reservoir", "bulk_total", "bulk+top"}:
                removal = removal_reservoir_exchange
            elif removal_key in {"interface_diff", "interface_flux", "interface"}:
                removal = removal_interface_diff
            elif removal_key in {"interface_total", "interface_total_flux"}:
                removal = removal_interface_total
            else:
                raise ValueError(
                    f"Unknown --removal-metric={args.removal_metric!r}. "
                    "Use 'consumption', 'top_flux', 'bulk_exchange', 'reservoir_exchange', 'interface_diff', or 'interface_total'."
                )

            # Case 5: coupled completely-mixed bulk ODE (COD only, simplified).
            # We use the conservation form:
            #   dS_bulk/dt = (Q(S_in - S_bulk) - A_L * removal) / V_L
            # with V_L = V_R - A_L (L + L_L). This neglects explicit external mass-transfer
            # resistance; it provides a reduced analogue of Warner case 5 for the one-domain
            # COD-only mapping.
            if int(cid) == 5:
                Q = float(getattr(args, "case5_Q", 0.0) or 0.0)
                VR = float(getattr(args, "case5_VR", 0.0) or 0.0)
                AL = float(getattr(args, "case5_AL", 0.0) or 0.0)
                LL = float(getattr(args, "case5_LL", 0.0) or 0.0)
                if VR <= 0.0:
                    raise ValueError(f"case5_VR must be > 0; got {VR}.")
                if AL < 0.0 or Q < 0.0 or LL < 0.0:
                    raise ValueError(f"case5 parameters must be non-negative; got Q={Q}, AL={AL}, LL={LL}.")
                L_phys = float(L_ref_m) * float(L_eff_nondim)
                VL = VR - AL * (L_phys + LL)
                if VL <= 0.0:
                    raise RuntimeError(f"Case 5: reactor liquid volume became non-positive (VL={VL}).")
                S_in = float(args.S_high)
                dt_step = float(getattr(dt_c, "value", float(args.dt)))
                if dt_step <= 0.0:
                    dt_step = float(args.dt)
                a = dt_step * Q / VL
                b = dt_step * (Q * S_in - AL * removal) / VL
                S_next = (float(S_bulk_value) + b) / (1.0 + a)
                if not np.isfinite(S_next):
                    S_next = float(S_bulk_value)
                S_bulk_value = float(max(S_next, 0.0))
                S_bulk_c.value = float(S_bulk_value)

            # Update detachment coefficient for the *next* step (lagged usage).
            D_det_prev.value = _detachment_coeff_case(
                case_id=cid,
                t_days=t_k,
                dt_days=float(dt_step),
                L_thickness_nondim=L_thickness_nondim,
                L_ref_m=float(L_ref_m),
                eps_det_nondim=float(eps_det_eff),
                lambda_shear=float(args.lambda_shear),
                slough_mode=str(args.slough_mode),
                slough_drop_nondim=float(args.slough_drop_um) * 1.0e-6 / float(L_ref_m),
            )

            mu_extra_used = float(getattr(mu_extra_c, "value", 0.0))
            rows.append(
                {
                    "t_days": float(t_k),
                    "L_eff_um": float(L_eff_um),
                    "L_half_um": float(L_half_um),
                    "L_thickness_um": float(L_thickness_um),
                    "removal": float(removal),
                    "removal_top_flux": float(removal_top_flux),
                    "removal_bulk_exchange": float(removal_bulk_exchange),
                    "removal_surface_exchange": float(removal_surface_exchange),
                    "removal_reservoir_exchange": float(removal_reservoir_exchange),
                    "removal_consumption": float(removal_consumption),
                    "removal_interface_diff_grad": float(removal_interface_diff_grad) if np.isfinite(removal_interface_diff_grad) else float("nan"),
                    "removal_interface_diff_int": float(removal_interface_diff_int),
                    "removal_interface_diff": float(removal_interface_diff),
                    "removal_interface_total": float(removal_interface_total),
                    "uL_um_per_d": float(1.0e6 * float(uL_m_per_d)),
                    "D_det_prev": float(D_det_prev.value),
                    "k_g_eff": float(k_g_used),
                    "S_surf": float(S_surf) if np.isfinite(float(S_surf)) else float("nan"),
                    "S_top": float(S_bulk_value),
                    "mu_extra": float(mu_extra_used),
                }
            )
            _write_rows_csv(rows=rows, out_csv=out_csv)

            # Case 2: update the effective COD-independent expansion term (for next step).
            mu_extra_base = float(getattr(args, "mu_extra_const", 0.0) or 0.0)
            if int(cid) == 2:
                mu_extra_c.value = mu_extra_base + _case2_mu_extra(
                    t_days=float(t_k),
                    enabled=bool(getattr(args, "case2_autotroph", False)),
                    start_days=float(getattr(args, "case2_autotroph_start", 6.0)),
                    mu0=float(getattr(args, "case2_autotroph_mu0", 0.0)),
                    mu1=float(getattr(args, "case2_autotroph_mu1", 0.0)),
                    tau_days=float(getattr(args, "case2_autotroph_tau", 1.0)),
                )
            else:
                mu_extra_c.value = mu_extra_base

            # Update thickness-scaled growth coefficient for the *next* step.
            if k_g_mode == "warner" and k_g_base != 0.0:
                A_01 = assemble_scalar(dh, I_alpha01, backend=backend, quad_order=int(qdeg))
                I01 = float(A_01) / max(width, 1.0e-16)
                k_g_c.value = float(k_g_base) * (float(L_eff_nondim) / max(float(I01), 1.0e-16))

            if bool(args.diagnostics):
                phi_k = _funcs[4]
                v_k = _funcs[0]
                p_k = _funcs[1]
                vS_k = _funcs[2]
                u_k = _funcs[3]
                a_min = float(np.min(alpha_k.nodal_values))
                a_max = float(np.max(alpha_k.nodal_values))
                p_min = float(np.min(phi_k.nodal_values))
                p_max = float(np.max(phi_k.nodal_values))
                s_min = float(np.min(S_k.nodal_values))
                s_max = float(np.max(S_k.nodal_values))
                vmax = float(np.max(np.abs(v_k.nodal_values))) if hasattr(v_k, "nodal_values") else float("nan")
                pmax = float(np.max(np.abs(p_k.nodal_values))) if hasattr(p_k, "nodal_values") else float("nan")
                vSmax = float(np.max(np.abs(vS_k.nodal_values))) if hasattr(vS_k, "nodal_values") else float("nan")
                umax = float(np.max(np.abs(u_k.nodal_values))) if hasattr(u_k, "nodal_values") else float("nan")
                print(
                    f"[diag] t={t_k:.3f}d  L_eff={L_eff_um:.3f}um  L_thick={L_thickness_um:.3f}um  removal={removal:.3e}  "
                    f"alpha=[{a_min:.3e},{a_max:.3e}]  phi=[{p_min:.3e},{p_max:.3e}]  S=[{s_min:.3e},{s_max:.3e}]  "
                    f"|v|_max={vmax:.3e}  |p|_max={pmax:.3e}  |vS|_max={vSmax:.3e}  |u|_max={umax:.3e}"
                )

        def _post_cb(_funcs):
            solver_i = solver_ref.get("solver")
            if solver_i is None:
                return
            t_k = float(solver_i._current_t + solver_i._current_dt)
            _record_step(t_k=t_k, _funcs=_funcs)

        use_bounds = bool(getattr(args, "pdas", True)) and bool(getattr(args, "bounds", True))
        newton_params = NewtonParameters(
            newton_tol=float(args.newton_tol),
            newton_rtol=float(getattr(args, "newton_rtol", 0.0) or 0.0),
            max_newton_iter=int(args.max_it),
            ls_mode=str(args.ls_mode),
        )
        newton_params.line_search = bool(getattr(args, "line_search", True))
        strategy = str(getattr(args, "solve_strategy", "monolithic")).strip().lower()
        if strategy not in {"monolithic", "split"}:
            raise ValueError(f"Unknown --solve-strategy={args.solve_strategy!r}. Use 'monolithic' or 'split'.")

        def _new_solver(res_form, jac_form, *, bounds_by_field: dict | None, post_cb=None):
            if use_bounds:
                os.environ.setdefault("PYCUTFEM_LS_FAIL_HARD", "0")
                solver_i = PdasNewtonSolver(
                    res_form,
                    jac_form,
                    dof_handler=dh,
                    mixed_element=prob["me"],
                    bcs=bcs,
                    bcs_homog=bcs_homog,
                    vi_params=VIParameters(
                        c=float(getattr(args, "vi_c", 0.0) or 0.0),
                        active_tol=float(getattr(args, "vi_active_tol", 0.0) or 0.0),
                        project_initial_guess=True,
                        project_each_iteration=bool(getattr(args, "vi_project_each_iter", False)),
                    ),
                    newton_params=newton_params,
                    quad_order=int(qdeg),
                    backend=backend,
                    postproc_timeloop_cb=post_cb,
                )
                if bounds_by_field:
                    solver_i.set_box_bounds(by_field=bounds_by_field)
                return solver_i
            return NewtonSolver(
                res_form,
                jac_form,
                dof_handler=dh,
                mixed_element=prob["me"],
                bcs=bcs,
                bcs_homog=bcs_homog,
                newton_params=newton_params,
                quad_order=int(qdeg),
                backend=backend,
                postproc_timeloop_cb=post_cb,
            )

        if strategy == "monolithic":
            residual_form = base_residual_form
            jacobian_form = base_jacobian_form
            if bulk_residual is not None and bulk_jacobian is not None:
                residual_form = residual_form + bulk_residual
                jacobian_form = jacobian_form + bulk_jacobian
            if surface_residual is not None and surface_jacobian is not None:
                residual_form = residual_form + surface_residual
                jacobian_form = jacobian_form + surface_jacobian
            solver = _new_solver(
                residual_form,
                jacobian_form,
                bounds_by_field={"alpha": (0.0, 1.0), "S": (0.0, None), "phi": (0.0, 1.0)} if use_bounds else None,
                post_cb=_post_cb,
            )
            solver_ref["solver"] = solver
            solver_alpha = solver
        else:
            # Split solve:
            #   (A) solve all blocks except substrate with S frozen to S_n,
            #   (B) solve substrate with all other fields frozen to (A).
            S_test = TestFunction("S", dof_handler=dh)
            dS_trial = TrialFunction("S", dof_handler=dh)

            freeze_scale = Constant(1.0)
            r_freeze_Sn = freeze_scale * (S_k - S_n) * S_test * dx_q
            a_freeze_Sn = freeze_scale * dS_trial * S_test * dx_q

            res_A = base_residual_form - forms.r_substrate + r_freeze_Sn
            jac_A = base_jacobian_form - forms.a_substrate + a_freeze_Sn

            # Hold copies for freezing in the substrate stage.
            v_hold = VectorFunction("v_hold", ["v_x", "v_y"], dof_handler=dh)
            vS_hold = VectorFunction("vS_hold", ["vS_x", "vS_y"], dof_handler=dh)
            u_hold = VectorFunction("u_hold", ["u_x", "u_y"], dof_handler=dh)
            p_hold = Function("p_hold", "p", dof_handler=dh)
            phi_hold = Function("phi_hold", "phi", dof_handler=dh)
            alpha_hold = Function("alpha_hold", "alpha", dof_handler=dh)
            lambda_hold = Function("lambda_alpha_hold", "lambda_alpha", dof_handler=dh) if "lambda_alpha" in fields else None

            V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
            VS = FunctionSpace("VS", ["vS_x", "vS_y"], dim=1)
            U = FunctionSpace("U", ["u_x", "u_y"], dim=1)
            v_test = VectorTestFunction(space=V, dof_handler=dh)
            vS_test = VectorTestFunction(space=VS, dof_handler=dh)
            u_test = VectorTestFunction(space=U, dof_handler=dh)
            dv = VectorTrialFunction(space=V, dof_handler=dh)
            dvS = VectorTrialFunction(space=VS, dof_handler=dh)
            du = VectorTrialFunction(space=U, dof_handler=dh)
            q_test = TestFunction("p", dof_handler=dh)
            dp = TrialFunction("p", dof_handler=dh)
            phi_test = TestFunction("phi", dof_handler=dh)
            dphi = TrialFunction("phi", dof_handler=dh)
            alpha_test = TestFunction("alpha", dof_handler=dh)
            dalpha = TrialFunction("alpha", dof_handler=dh)

            r_freeze_other = freeze_scale * dot(fields["v"] - v_hold, v_test) * dx_q
            a_freeze_other = freeze_scale * dot(dv, v_test) * dx_q
            r_freeze_other += freeze_scale * (fields["p"] - p_hold) * q_test * dx_q
            a_freeze_other += freeze_scale * dp * q_test * dx_q
            r_freeze_other += freeze_scale * dot(fields["vS"] - vS_hold, vS_test) * dx_q
            a_freeze_other += freeze_scale * dot(dvS, vS_test) * dx_q
            r_freeze_other += freeze_scale * dot(fields["u"] - u_hold, u_test) * dx_q
            a_freeze_other += freeze_scale * dot(du, u_test) * dx_q
            r_freeze_other += freeze_scale * (fields["phi"] - phi_hold) * phi_test * dx_q
            a_freeze_other += freeze_scale * dphi * phi_test * dx_q
            r_freeze_other += freeze_scale * (fields["alpha"] - alpha_hold) * alpha_test * dx_q
            a_freeze_other += freeze_scale * dalpha * alpha_test * dx_q

            if lambda_hold is not None:
                lambda_test = TestFunction("lambda_alpha", dof_handler=dh)
                dlambda = TrialFunction("lambda_alpha", dof_handler=dh)
                r_freeze_other += freeze_scale * (fields["lambda_alpha"] - lambda_hold) * lambda_test * dx_q
                a_freeze_other += freeze_scale * dlambda * lambda_test * dx_q

            res_B = forms.r_substrate
            jac_B = forms.a_substrate
            if bulk_residual is not None and bulk_jacobian is not None:
                res_B = res_B + bulk_residual
                jac_B = jac_B + bulk_jacobian
            if surface_residual is not None and surface_jacobian is not None:
                res_B = res_B + surface_residual
                jac_B = jac_B + surface_jacobian
            res_B = res_B + r_freeze_other
            jac_B = jac_B + a_freeze_other

            solver_A = _new_solver(res_A, jac_A, bounds_by_field={"alpha": (0.0, 1.0), "phi": (0.0, 1.0)} if use_bounds else None)
            solver_B = _new_solver(res_B, jac_B, bounds_by_field={"S": (0.0, None)} if use_bounds else None)

            solver_ref["solver"] = solver_B
            solver_alpha = solver_A

            # Optional substrate warmup: with thick-film restarts (t_start>0, h0>>eps0),
            # a split strategy can fail because stage (A) uses the *frozen* substrate
            # state. Warm up S by repeatedly solving only the substrate stage with all
            # other fields frozen to the current state.
            warm_steps = int(getattr(args, "warmup_substrate_steps", 0) or 0)
            warm_dt = float(getattr(args, "warmup_substrate_dt", 0.0) or 0.0)
            if warm_steps > 0:
                if not (math.isfinite(warm_dt) and warm_dt > 0.0):
                    raise ValueError(f"--warmup-substrate-dt must be > 0; got {getattr(args, 'warmup_substrate_dt', None)!r}.")
                warm_steps = int(max(warm_steps, 0))
                t_warm0 = float(max(0.0, float(t_start) - float(warm_steps) * float(warm_dt)))
                if int(cid) != 5:
                    # Ensure the bulk reservoir value is consistent at the warmup start time.
                    S_bulk_value = float(_S_bulk_case(cid, float(t_warm0), S_high=float(args.S_high)))
                    S_bulk_c.value = float(S_bulk_value)

                aux_B_warm = {
                    "dt": dt_c,
                    "v_hold": v_hold,
                    "p_hold": p_hold,
                    "vS_hold": vS_hold,
                    "u_hold": u_hold,
                    "phi_hold": phi_hold,
                    "alpha_hold": alpha_hold,
                }
                if lambda_hold is not None:
                    aux_B_warm["lambda_alpha_hold"] = lambda_hold

                for j in range(int(warm_steps)):
                    t_n_w = float(t_warm0 + float(j) * float(warm_dt))
                    dt_c.value = float(warm_dt)

                    # Refresh holds from the previous state and reset the predictor.
                    for f, f_prev in zip(funcs, prev_funcs):
                        f.nodal_values[:] = f_prev.nodal_values[:]
                    v_hold.nodal_values[:] = fields["v"].nodal_values[:]
                    vS_hold.nodal_values[:] = fields["vS"].nodal_values[:]
                    u_hold.nodal_values[:] = fields["u"].nodal_values[:]
                    p_hold.nodal_values[:] = fields["p"].nodal_values[:]
                    phi_hold.nodal_values[:] = fields["phi"].nodal_values[:]
                    alpha_hold.nodal_values[:] = fields["alpha"].nodal_values[:]
                    if lambda_hold is not None:
                        lambda_hold.nodal_values[:] = fields["lambda_alpha"].nodal_values[:]

                    # Update bulk substrate value for the warmup time (case 2 step-change).
                    if int(cid) != 5:
                        S_bulk_value = float(_S_bulk_case(cid, float(t_n_w), S_high=float(args.S_high)))
                        S_bulk_c.value = float(S_bulk_value)

                    t_bc_w = float(t_n_w + float(theta) * float(warm_dt))
                    bcs_now_w = solver_B._freeze_bcs(bcs, t_bc_w)
                    dh.apply_bcs(bcs_now_w, *funcs)

                    solver_B._current_step_no = int(-1000000 + j)
                    solver_B._current_t = float(t_n_w)
                    solver_B._current_dt = float(warm_dt)
                    solver_B._newton_loop(funcs, prev_funcs, aux_B_warm, bcs_now_w)
                    dh.apply_bcs(bcs_now_w, *funcs)

                    # Accept warmup step: sync previous state.
                    for f, f_prev in zip(funcs, prev_funcs):
                        f_prev.nodal_values[:] = f.nodal_values[:]

                # After warmup, ensure current and previous alpha are consistent for any
                # later in-step "shift" sloughing updates.
                alpha_n.nodal_values[:] = alpha_k.nodal_values[:]

        # Conservative Allen–Cahn (eliminated λ): enforce
        #   ∫ M(α) (μ_α - λ_α) dx = 0
        # by projecting λ_α from α before each assembly, using the same by-parts
        # form as `build_biofilm_one_domain_forms`.
        alpha_cahn_conservative = bool(getattr(args, "alpha_cahn_conservative", False))
        cons_mode = str(getattr(args, "alpha_cahn_conservative_mode", "eliminate")).strip().lower()
        solve_lambda = alpha_cahn_conservative and cons_mode == "unknown"
        lambda_alpha_k = prob.get("lambda_alpha_k", None)
        if alpha_cahn_conservative and (not solve_lambda) and (lambda_alpha_k is not None):
            qdeg_i = int(qdeg)

            def _update_lambda_alpha_from_alpha(_coeffs: dict[str, object]) -> None:
                gamma_ac = float(getattr(args, "alpha_cahn_gamma", 0.0) or 0.0)
                eps_ac_eff = max(float(eps_ac), 1.0e-16)
                if gamma_ac == 0.0:
                    lambda_alpha_k.nodal_values[:] = 0.0
                    return

                mob_key = str(getattr(args, "alpha_cahn_mobility", "constant")).strip().lower()
                if mob_key in {"constant", "const"}:
                    mob = Constant(1.0)
                    mob_prime = Constant(0.0)
                else:
                    mob = alpha_k * (Constant(1.0) - alpha_k)
                    mob_prime = Constant(1.0) - Constant(2.0) * alpha_k

                den = float(assemble_scalar(dh, mob * dx_q, backend=backend, quad_order=qdeg_i))
                if not np.isfinite(den) or den <= 1.0e-16:
                    lambda_alpha_k.nodal_values[:] = 0.0
                    return

                Wp = Constant(2.0) * alpha_k * (Constant(1.0) - alpha_k) * (Constant(1.0) - Constant(2.0) * alpha_k)
                num = float(assemble_scalar(dh, (mob * Wp) * dx_q, backend=backend, quad_order=qdeg_i))
                lam = (gamma_ac / eps_ac_eff) * (num / den)
                if mob_key not in {"constant", "const"}:
                    g2 = inner(grad(alpha_k), grad(alpha_k))
                    num2 = float(assemble_scalar(dh, (mob_prime * g2) * dx_q, backend=backend, quad_order=qdeg_i))
                    lam += (eps_ac_eff * gamma_ac) * (num2 / den)

                if not np.isfinite(lam):
                    lam = 0.0
                lambda_alpha_k.nodal_values[:] = float(lam)

            solver_alpha.preassemble_cb = _update_lambda_alpha_from_alpha

        def _on_dt_change(new_dt: float) -> None:
            dt_c.value = float(new_dt)

        # Set initial detachment coefficient for t=t_start.
        D_det_prev.value = _detachment_coeff_case(
            case_id=cid,
            t_days=float(t_start),
            dt_days=float(args.dt),
            L_thickness_nondim=float(args.h0),
            L_ref_m=float(L_ref_m),
            eps_det_nondim=float(eps_det_eff),
            lambda_shear=float(args.lambda_shear),
            slough_mode=str(args.slough_mode),
            slough_drop_nondim=float(args.slough_drop_um) * 1.0e-6 / float(L_ref_m),
        )

        if strategy == "monolithic":
            dt_max_val = float(args.dt) if float(getattr(args, "dt_max", 0.0) or 0.0) <= 0.0 else float(args.dt_max)

            # Split the long run at discontinuous "event" times so we hit them exactly
            # (and can restart dt for the post-event transient).
            t_start = float(getattr(args, "t_start", 0.0) or 0.0)
            t_final = float(args.t_final)
            t_events: list[float] = []
            if int(cid) == 2:
                t_events.append(6.0)  # step-change in surface substrate
            if int(cid) == 4:
                slough_mode = str(getattr(args, "slough_mode", "integrated")).strip().lower()
                if slough_mode in {"exact", "shift", "shift_window"}:
                    t_events.extend([5.984, 5.994])  # Warner window
                else:
                    t_events.append(6.0)  # integrated coarse drop around 6d

            dt_base = float(args.dt)
            # For discontinuities, also stop one base step before the event so the
            # (event-ε, event] interval can be taken with a smaller restarted dt.
            if int(cid) in {2, 4}:
                for te in list(t_events):
                    t_pre = float(te) - float(dt_base)
                    if float(t_start) < t_pre < t_final:
                        t_events.append(t_pre)

            # Keep only strict interior events; always end at t_final.
            t_stops = sorted({t for t in t_events if float(t_start) < float(t) < t_final})
            t_stops.append(t_final)

            t0 = float(t_start)
            first_segment = True
            dt_after = float(getattr(args, "dt_after_event", 0.0) or 0.0)
            if dt_after <= 0.0:
                dt_after = float(dt_base)

            for t1 in t_stops:
                if t1 <= t0 + 1.0e-15:
                    t0 = float(t1)
                    continue

                dt_init = float(dt_base) if first_segment else float(min(dt_base, dt_after))
                solver.solve_time_interval(
                    functions=funcs,
                    prev_functions=prev_funcs,
                    aux_functions={"dt": dt_c},
                    time_params=TimeStepperParameters(
                        dt=float(dt_init),
                        final_time=float(t1),
                        max_steps=int(1.0e9),
                        theta=float(theta),
                        t0=float(t0),
                        stop_on_steady=False,
                        on_dt_change=_on_dt_change,
                        allow_dt_reduction=bool(getattr(args, "adaptive_dt", False)),
                        dt_min=float(getattr(args, "dt_min", 0.0) or 0.0),
                        dt_max=float(dt_max_val),
                        dt_reduction_factor=float(getattr(args, "dt_reduction_factor", 0.5) or 0.5),
                        dt_increase_factor=float(getattr(args, "dt_increase_factor", 2.0) or 2.0),
                        dt_iters_increase_threshold=int(getattr(args, "dt_iters_increase_threshold", 25) or 25),
                        dt_easy_steps_before_increase=int(getattr(args, "dt_easy_steps_before_increase", 1) or 1),
                        dt_decrease_factor_slow=float(getattr(args, "dt_decrease_factor_slow", 1.0) or 1.0),
                        dt_iters_decrease_threshold=int(getattr(args, "dt_iters_decrease_threshold", 40) or 40),
                        dt_slow_steps_before_decrease=int(getattr(args, "dt_slow_steps_before_decrease", 1) or 1),
                        dt_reject_on_slow=bool(getattr(args, "dt_reject_on_slow", False)),
                    ),
                )
                t0 = float(t1)
                first_segment = False
        else:
            # Manual time loop for split strategy.
            #
            # IMPORTANT: cases 2 and 4 have discontinuous "events" at ~t=6d. We must
            # hit these times exactly (same as the monolithic driver) so the frozen
            # θ-time BC evaluation does not smear the discontinuity across a step.
            t_start = float(getattr(args, "t_start", 0.0) or 0.0)
            t_final = float(args.t_final)
            dt_base = float(args.dt)
            dt_max_val = float(args.dt) if float(getattr(args, "dt_max", 0.0) or 0.0) <= 0.0 else float(args.dt_max)
            dt_min_val = float(getattr(args, "dt_min", 0.0) or 0.0)
            allow_adapt = bool(getattr(args, "adaptive_dt", False))
            dt_reduction = float(getattr(args, "dt_reduction_factor", 0.5) or 0.5)
            dt_increase = float(getattr(args, "dt_increase_factor", 2.0) or 2.0)

            # Split the run at discontinuous "event" times so we hit them exactly.
            t_events: list[float] = []
            if int(cid) == 2:
                t_events.append(6.0)  # step-change in surface substrate
            if int(cid) == 4:
                slough_mode = str(getattr(args, "slough_mode", "integrated")).strip().lower()
                if slough_mode in {"exact", "shift", "shift_window"}:
                    t_events.extend([5.984, 5.994])  # Warner window
                else:
                    t_events.append(6.0)  # integrated coarse drop around 6d

            # For discontinuities, also stop one base step before the event so the
            # (event-Δt, event] interval can be taken with a restarted dt.
            if int(cid) in {2, 4}:
                for te in list(t_events):
                    t_pre = float(te) - float(dt_base)
                    if float(t_start) < t_pre < t_final:
                        t_events.append(t_pre)

            # Keep only strict interior events; always end at t_final.
            t_stops = sorted({t for t in t_events if float(t_start) < float(t) < t_final})
            t_stops.append(t_final)

            dt_after = float(getattr(args, "dt_after_event", 0.0) or 0.0)
            if dt_after <= 0.0:
                dt_after = float(dt_base)

            step_no = 0
            t0 = float(t_start)
            first_segment = True
            for t1 in t_stops:
                if t1 <= t0 + 1.0e-15:
                    t0 = float(t1)
                    continue

                # Restart dt after events (and ramp back up if adaptive dt is enabled).
                dt = float(dt_base) if first_segment else float(min(dt_base, dt_after))
                t_n = float(t0)
                first_segment = False

                # IMPORTANT: for case 4 "integrated" sloughing, the detachment
                # coefficient depends on the *upcoming* step length (to detect a
                # crossing of 6d). Because we restart dt at each segment boundary,
                # the lagged update inside `_record_step` can otherwise miss the
                # event if the last step of the previous segment had a smaller dt.
                dt_first = float(min(dt, t1 - t_n))
                if dt_first > 0.0:
                    thickness_key = str(getattr(args, "thickness_metric", "half")).strip().lower()
                    if mode == "strip":
                        try:
                            L_half_seg, _, _ = _strip_interface_metrics(dh=dh, alpha=alpha_k, S=S_k)
                        except Exception:
                            L_half_seg = _strip_thickness_alpha_half(dh=dh, alpha=alpha_k)

                        if thickness_key == "eff":
                            A_alpha = assemble_scalar(dh, I_alpha, backend=backend, quad_order=int(qdeg))
                            L_thickness_seg = float(A_alpha) / max(width, 1.0e-16)
                        else:
                            L_thickness_seg = float(L_half_seg)
                    else:
                        # 2D mode: use the initial thickness scale as a fallback.
                        L_thickness_seg = float(getattr(args, "h0", 0.0) or 0.0)

                    D_det_prev.value = _detachment_coeff_case(
                        case_id=cid,
                        t_days=float(t_n),
                        dt_days=float(dt_first),
                        L_thickness_nondim=float(L_thickness_seg),
                        L_ref_m=float(L_ref_m),
                        eps_det_nondim=float(eps_det_eff),
                        lambda_shear=float(args.lambda_shear),
                        slough_mode=str(args.slough_mode),
                        slough_drop_nondim=float(args.slough_drop_um) * 1.0e-6 / float(L_ref_m),
                    )

                while t_n < t1 - 1.0e-15:
                    dt_step = float(min(dt, t1 - t_n))
                    dt_c.value = float(dt_step)

                    # Predictor: reset to previous accepted state for each attempt.
                    for f, f_prev in zip(funcs, prev_funcs):
                        f.nodal_values[:] = f_prev.nodal_values[:]

                    # IMPORTANT: update any *time-dependent* imposed bulk values *before* the
                    # PDE solve so the step that hits an event time (e.g. case 2 at t=6d)
                    # is solved with the correct forcing.
                    t_bc = float(t_n + float(theta) * dt_step)
                    if int(cid) != 5:
                        S_bulk_value = float(_S_bulk_case(cid, float(t_bc), S_high=float(args.S_high)))
                        S_bulk_c.value = float(S_bulk_value)
                    mu_extra_base = float(getattr(args, "mu_extra_const", 0.0) or 0.0)
                    if int(cid) == 2:
                        mu_extra_c.value = mu_extra_base + _case2_mu_extra(
                            t_days=float(t_bc),
                            enabled=bool(getattr(args, "case2_autotroph", False)),
                            start_days=float(getattr(args, "case2_autotroph_start", 6.0)),
                            mu0=float(getattr(args, "case2_autotroph_mu0", 0.0)),
                            mu1=float(getattr(args, "case2_autotroph_mu1", 0.0)),
                            tau_days=float(getattr(args, "case2_autotroph_tau", 1.0)),
                        )
                    else:
                        mu_extra_c.value = mu_extra_base

                    # Case 4 "shift_window": apply sloughing *before* the PDE solve so the
                    # substrate/mechanics stages evolve on the updated interface position.
                    #
                    # If we apply the truncation after the solve (post-step), diagnostics at
                    # the window endpoint (t=5.994) are inconsistent (α is updated but S is
                    # still the pre-slough solution). Applying it pre-step avoids that.
                    slough_drop_now = 0.0
                    alpha_prev_backup = None
                    S_prev_backup = None
                    if int(cid) == 4 and slough_mode_key == "shift_window":
                        win0 = 5.984
                        win1 = 5.994
                        t0_step = float(t_n)
                        t1_step = float(t_n + dt_step)
                        overlap = max(0.0, min(t1_step, win1) - max(t0_step, win0))
                        remaining = float(slough_drop_nondim) - float(slough_shift_window_applied_nondim)
                        if overlap > 0.0 and remaining > 1.0e-15:
                            V_nondim = float(slough_drop_nondim) / float(win1 - win0)
                            slough_drop_now = float(min(float(V_nondim) * float(overlap), remaining))
                            if slough_drop_now > 0.0:
                                # Backup previous state so we can rollback if the nonlinear solve fails
                                # and we retry with a smaller dt (adaptive_dt).
                                alpha_prev_backup = np.array(alpha_n.nodal_values, copy=True)
                                S_prev_backup = np.array(S_n.nodal_values, copy=True)

                                _apply_slough_truncate_strip(
                                    dh=dh,
                                    alpha=alpha_n,
                                    drop_nondim=float(slough_drop_now),
                                    eps_nondim=float(args.eps0),
                                )
                                try:
                                    L_new = float(_strip_thickness_alpha_half(dh=dh, alpha=alpha_n))
                                    coords_S = np.asarray(dh.get_dof_coords("S"), dtype=float)
                                    if coords_S.ndim == 2 and coords_S.shape[1] >= 2 and coords_S.shape[0] == np.asarray(S_n.nodal_values).size:
                                        mask = coords_S[:, 1] >= float(L_new)
                                        S_n.nodal_values[:] = np.where(mask, float(S_bulk_value), S_n.nodal_values)
                                except Exception:
                                    pass

                                # Sync current predictor to the shifted previous state.
                                alpha_k.nodal_values[:] = alpha_n.nodal_values[:]
                                S_k.nodal_values[:] = S_n.nodal_values[:]

                    step_no_next = int(step_no + 1)
                    bcs_now = solver_A._freeze_bcs(bcs, t_bc)
                    dh.apply_bcs(bcs_now, *funcs)

                    try:
                        solver_A._current_step_no = int(step_no_next)
                        solver_A._current_t = float(t_n)
                        solver_A._current_dt = float(dt_step)
                        solver_A._newton_loop(funcs, prev_funcs, {"dt": dt_c}, bcs_now)
                        dh.apply_bcs(bcs_now, *funcs)

                        # Freeze targets for the substrate stage
                        v_hold.nodal_values[:] = fields["v"].nodal_values[:]
                        vS_hold.nodal_values[:] = fields["vS"].nodal_values[:]
                        u_hold.nodal_values[:] = fields["u"].nodal_values[:]
                        p_hold.nodal_values[:] = fields["p"].nodal_values[:]
                        phi_hold.nodal_values[:] = fields["phi"].nodal_values[:]
                        alpha_hold.nodal_values[:] = fields["alpha"].nodal_values[:]
                        if lambda_hold is not None:
                            lambda_hold.nodal_values[:] = fields["lambda_alpha"].nodal_values[:]

                        solver_B._current_step_no = int(step_no_next)
                        solver_B._current_t = float(t_n)
                        solver_B._current_dt = float(dt_step)
                        aux_B = {
                            "dt": dt_c,
                            # Hold fields referenced by the freezing terms in (B).
                            # Needed for the cpp backend so the kernel sees these coefficients.
                            "v_hold": v_hold,
                            "p_hold": p_hold,
                            "vS_hold": vS_hold,
                            "u_hold": u_hold,
                            "phi_hold": phi_hold,
                            "alpha_hold": alpha_hold,
                        }
                        if lambda_hold is not None:
                            aux_B["lambda_alpha_hold"] = lambda_hold
                        solver_B._newton_loop(funcs, prev_funcs, aux_B, bcs_now)
                        dh.apply_bcs(bcs_now, *funcs)
                    except Exception:
                        # Rollback pre-step sloughing update before retrying.
                        if alpha_prev_backup is not None and S_prev_backup is not None:
                            alpha_n.nodal_values[:] = alpha_prev_backup
                            S_n.nodal_values[:] = S_prev_backup
                        if not allow_adapt:
                            raise
                        dt = float(dt) * float(dt_reduction)
                        if dt_min_val > 0.0 and float(dt) < float(dt_min_val) - 1.0e-15:
                            raise
                        continue

            # Accepted step: record, advance time, update step counter.
                    # Snap times to the segment endpoint to avoid missing case-event
                    # windows due to roundoff (e.g. 5.984 represented as 5.983999999999999).
                    t_k_raw = float(t_n + dt_step)
                    t_k = float(t1) if abs(float(t_k_raw) - float(t1)) <= _TIME_TOL_DAYS else float(t_k_raw)
                    if float(slough_drop_now) > 0.0:
                        slough_shift_window_applied_nondim = float(slough_shift_window_applied_nondim) + float(slough_drop_now)
                    _record_step(t_k=t_k, _funcs=funcs)
                    for f, f_prev in zip(funcs, prev_funcs):
                        f_prev.nodal_values[:] = f.nodal_values[:]
                    t_n = float(t_k)
                    step_no = int(step_no_next)

                    if allow_adapt:
                        dt = float(min(float(dt_max_val), float(dt) * float(dt_increase)))

                t0 = float(t1)

        rows_by_mode_case.setdefault((mode, cid), []).extend(rows)

        # Write per-run CSV (final snapshot; partial snapshots are written per step).
        _write_rows_csv(rows=rows, out_csv=out_csv)
        return out_csv

    # ------------------------------------------------------------------
    # Run strip and/or 2D cases
    # ------------------------------------------------------------------
    solver_ref: dict[str, object] = {}

    strip_csv_by_case: dict[int, Path] = {}
    if args.mode in {"strip", "both"}:
        for cid in cases:
            strip_csv_by_case[cid] = _run_mode(
                "strip",
                cid=cid,
                Lx=float(args.Lx),
                Hy=float(args.Hy),
                nx=int(args.nx),
                ny=int(args.ny),
                h0_fn=None,
            )

    if args.mode in {"2d", "both"}:
        # Parse 2D cases (subset of {1,2,3,4}).
        cases_2d: list[int] = []
        for part in str(args.cases_2d).split(","):
            part = part.strip()
            if not part:
                continue
            cid = int(part)
            if cid not in {1, 2, 3, 4, 5}:
                raise ValueError(f"--cases-2d must be a subset of {{1,2,3,4,5}}; got {cid}.")
            cases_2d.append(cid)
        cases_2d = sorted(set(cases_2d))
        if not cases_2d:
            cases_2d = [1]

        Lx2 = float(args.Hy)  # use a square-ish domain by default (width ~ height)
        amp = float(args.h0_amp)

        def _h0_wavy(x: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=float)
            return float(args.h0) * (1.0 + amp * np.sin(2.0 * math.pi * xx / max(Lx2, 1.0e-12)))

        for cid_2d in cases_2d:
            _run_mode(
                "2d",
                cid=cid_2d,
                Lx=Lx2,
                Hy=float(args.Hy),
                nx=int(args.nx_2d),
                ny=int(args.ny_2d),
                h0_fn=_h0_wavy,
            )

    # ------------------------------------------------------------------
    # Plot overlays vs Warner reference
    # ------------------------------------------------------------------
    if not bool(args.no_plots):
        import matplotlib

        if not os.getenv("DISPLAY", ""):
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        for cid in cases:
            rows = rows_by_mode_case.get(("strip", cid), [])
            if not rows:
                continue
            r = ref_by_case[cid]
            t = np.asarray([rr["t_days"] for rr in rows], dtype=float)
            thickness_key = str(getattr(args, "thickness_metric", "half")).strip().lower()
            if thickness_key == "eff":
                L_eff = np.asarray([rr["L_eff_um"] for rr in rows], dtype=float)
                thickness_label = "L_eff"
            else:
                L_eff = np.asarray([rr["L_half_um"] for rr in rows], dtype=float)
                thickness_label = "L_{α=0.5}"
            rem = np.asarray([rr["removal"] for rr in rows], dtype=float)

            fig, ax = plt.subplots(2, 1, figsize=(6.5, 6.0), sharex=True)
            ax[0].plot(r.t_days, r.L_um, "k-", label="Warner1986 (FD)")
            ax[0].plot(t, L_eff, "C0-o", ms=3, label=f"one-domain ({thickness_label}, {args.mode})")
            ax[0].set_ylabel("thickness [µm]")
            ax[0].grid(True, alpha=0.3)
            ax[0].legend(loc="best", fontsize=9)

            ax[1].plot(r.t_days, r.removal, "k-", label="Warner1986 (FD)")
            ax[1].plot(t, rem, "C1-o", ms=3, label="one-domain")
            ax[1].set_xlabel("t [d]")
            ax[1].set_ylabel("COD removal [g m$^{-2}$ d$^{-1}$]")
            ax[1].grid(True, alpha=0.3)
            ax[1].legend(loc="best", fontsize=9)

            fig.suptitle(f"Warner1986 case {cid}: one-domain strip comparison (backend={backend})")
            fig.tight_layout()
            fig.savefig(outdir / f"one_domain_case{cid}_compare_backend={backend}.png", dpi=200)
            if paper_figdir is not None:
                fig.savefig(paper_figdir / f"warner1986_one_domain_case{cid}_compare_backend={backend}.pdf")
            plt.close(fig)

        # If a 2D run was requested, compare 2D averages to the strip baseline (for cases available in both).
        for cid in cases:
            rows_strip = rows_by_mode_case.get(("strip", cid), [])
            rows_2d = rows_by_mode_case.get(("2d", cid), [])
            if not (rows_strip and rows_2d):
                continue
            t1 = np.asarray([rr["t_days"] for rr in rows_strip], dtype=float)
            L1 = np.asarray([rr["L_eff_um"] for rr in rows_strip], dtype=float)
            j1 = np.asarray([rr["removal"] for rr in rows_strip], dtype=float)
            t2 = np.asarray([rr["t_days"] for rr in rows_2d], dtype=float)
            L2 = np.asarray([rr["L_eff_um"] for rr in rows_2d], dtype=float)
            j2 = np.asarray([rr["removal"] for rr in rows_2d], dtype=float)

            fig, ax = plt.subplots(2, 1, figsize=(6.5, 6.0), sharex=True)
            ax[0].plot(t1, L1, "C0-o", ms=3, label="strip (avg)")
            ax[0].plot(t2, L2, "C2-o", ms=3, label="2D (avg)")
            ax[0].set_ylabel("thickness [µm]")
            ax[0].grid(True, alpha=0.3)
            ax[0].legend(loc="best", fontsize=9)

            ax[1].plot(t1, j1, "C1-o", ms=3, label="strip (avg)")
            ax[1].plot(t2, j2, "C3-o", ms=3, label="2D (avg)")
            ax[1].set_xlabel("t [d]")
            ax[1].set_ylabel("COD removal [g m$^{-2}$ d$^{-1}$]")
            ax[1].grid(True, alpha=0.3)
            ax[1].legend(loc="best", fontsize=9)

            fig.suptitle(f"One-domain: 2D vs strip averages (case {cid}, backend={backend})")
            fig.tight_layout()
            fig.savefig(outdir / f"one_domain_case{cid}_2d_vs_strip_backend={backend}.png", dpi=200)
            if paper_figdir is not None:
                fig.savefig(paper_figdir / f"warner1986_one_domain_case{cid}_2d_vs_strip_backend={backend}.pdf")
            plt.close(fig)


if __name__ == "__main__":
    main()
