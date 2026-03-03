"""
Duddu et al. (2009) detachment benchmark (1D reduction) for the one-domain biofilm model.

Reference
---------
R. Duddu, D. L. Chopp, B. Moran (2009)
"A Two-Dimensional Continuum Model of Biofilm Growth Incorporating Fluid Flow and Shear
 Stress Based Detachment", Biotechnol. Bioeng. 103:92–104.

Aligned numerical example (from main.tex, Detachment Model Suggested in Rittman (1982))
-------------------------------------------------------------------------------------
The paper derives, for a 1D biofilm in a half-channel of height H, a shear-based detachment
law F_det = a |tau|^b with |tau| = 2 mu u_max^0 H / (H-l)^2, and shows for b=1/2:
  F_det(l) = k (1 - l/H)^(-1) = k (1 + l/H + (l/H)^2 + ...),
where l is the film thickness.

What this script verifies
-------------------------
1) The one-domain model's interface sink coefficient D_det_prev maps to a *physical*
   interface recession speed via:
       V_det ≈ 4 eps_alpha * D_det_prev,
   consistent with examples/biofilms/model/model.tex (Eq. rate_from_speed).
2) For a 1D-like strip (uniform in x), the thickness evolution from the alpha PDE matches
   the thickness ODE:
       dl/dt = vS_y(l) - F_det(l),
   where we prescribe vS_y = g*y (growth-driven vertical stretching).

Notes
-----
This benchmark intentionally avoids the full poroelastic/fluid blocks by:
  - prescribing vS and holding all other fields fixed,
  - solving only the alpha equation (with optional Allen–Cahn regularization),
so the comparison focuses on the detachment-speed mapping and the 1D reduction used in
the paper.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import (
    HAS_PETSC,
    LinearSolverParameters,
    NewtonParameters,
    PdasNewtonSolver,
    TimeStepperParameters,
    VIParameters,
)
from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad

from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms


def _tag_rectangle_boundaries(mesh: Mesh, *, L: float, H: float, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - float(L)) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - float(H)) <= tol,
        }
    )


def _mark_inactive_fields(dh: DofHandler, *field_names: str) -> None:
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
    return 0.5 * (1.0 + np.tanh(np.asarray(z, dtype=float)))


def _alpha_step_eval(x: np.ndarray, y: np.ndarray, *, h: float, eps: float) -> np.ndarray:
    """Diffuse Heaviside: alpha≈1 for y<h, alpha≈0 for y>h."""
    eps = max(float(eps), 1.0e-12)
    xx, yy = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return _smooth_step((float(h) - yy) / eps)


def _as_float_time(fn):
    return lambda x, y, t: float(np.asarray(fn(np.asarray(x), np.asarray(y), float(t))).reshape(()))


def _strip_thickness_alpha_half(
    *,
    dh: DofHandler,
    alpha: Function,
    y_round: int = 12,
    alpha_half: float = 0.5,
) -> float:
    """
    Thickness estimate: locate y where x-averaged alpha crosses alpha_half.

    Returns thickness in the same units as the mesh coordinates (here: mm).
    """
    coords = np.asarray(dh.get_dof_coords("alpha"), dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        return float("nan")
    y = coords[:, 1]
    a = np.asarray(alpha.nodal_values, dtype=float).ravel()
    if y.size != a.size:
        return float("nan")

    y_key = np.round(y, decimals=int(y_round))
    y_levels, inv = np.unique(y_key, return_inverse=True)
    counts = np.bincount(inv)
    sums = np.bincount(inv, weights=a)
    with np.errstate(divide="ignore", invalid="ignore"):
        a_bar = np.where(counts > 0, sums / counts, np.nan)

    order = np.argsort(y_levels)
    y_levels = y_levels[order]
    a_bar = a_bar[order]

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


@dataclass(frozen=True)
class DudduRittmanParams:
    # Flow/viscosity parameters used in the paper's 1D derivation.
    H_mm: float = 0.5
    mu_Pa_s: float = 0.001
    u_max0_mm_s: float = 12.5
    a_mm_per_day_per_sqrt_Pa: float = 0.1
    b: float = 0.5


def _shear_stress_interface_Pa(*, l_mm: float, p: DudduRittmanParams) -> float:
    """
    Interface shear magnitude from the paper (1D half-channel Poiseuille model):
        |tau| = 2 mu u_max0 H / (H-l)^2
    with consistent length units for u and y (mm here).
    """
    H = float(p.H_mm)
    l = float(l_mm)
    if l >= H:
        return float("inf")
    denom = (H - l) ** 2
    return float(2.0 * float(p.mu_Pa_s) * float(p.u_max0_mm_s) * H / denom)


def detachment_speed_shear_mm_per_day(*, l_mm: float, p: DudduRittmanParams) -> float:
    tau = _shear_stress_interface_Pa(l_mm=float(l_mm), p=p)
    if not math.isfinite(tau) or tau <= 0.0:
        return 0.0
    return float(p.a_mm_per_day_per_sqrt_Pa) * float(tau ** float(p.b))


def detachment_speed_poly_mm_per_day(*, l_mm: float, p: DudduRittmanParams, order: int = 2) -> float:
    """
    Polynomial approximation of the b=1/2 shear law:
        F_det(l) = k (1 - l/H)^(-1) ≈ k Σ_{n=0..order} (l/H)^n
    with
        k = a sqrt(2 mu u_max0 / H).
    """
    if abs(float(p.b) - 0.5) > 1.0e-14:
        raise ValueError("Polynomial detachment approximation is only valid for b=1/2.")
    order = int(order)
    if order < 0:
        raise ValueError("order must be >= 0")

    H = float(p.H_mm)
    k = float(p.a_mm_per_day_per_sqrt_Pa) * float(math.sqrt((2.0 * float(p.mu_Pa_s) * float(p.u_max0_mm_s)) / H))
    r = float(l_mm) / H
    s = 0.0
    term = 1.0
    for _ in range(order + 1):
        s += term
        term *= r
    return float(k * s)


def detachment_speed_l2_mm_per_day(*, l_mm: float, k_l2: float) -> float:
    return float(k_l2) * float(l_mm) * float(l_mm)


def _rk4_step(f, y0: float, t0: float, dt: float) -> float:
    k1 = float(f(t0, y0))
    k2 = float(f(t0 + 0.5 * dt, y0 + 0.5 * dt * k1))
    k3 = float(f(t0 + 0.5 * dt, y0 + 0.5 * dt * k2))
    k4 = float(f(t0 + dt, y0 + dt * k3))
    return float(y0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: float(r[k]) for k in keys})


def main() -> None:
    ap = argparse.ArgumentParser(description="Duddu2009 1D detachment benchmark for the one-domain alpha equation.")
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--linear-solver", type=str, default=("petsc" if HAS_PETSC else "scipy"), choices=("petsc", "scipy"))
    ap.add_argument("--outdir", type=str, default="examples/biofilms/benchmarks/dadu/results/duddu2009_detachment_1d")

    # Geometry / mesh (mm)
    ap.add_argument("--Lx", type=float, default=0.2, help="Strip width (mm).")
    ap.add_argument("--H", type=float, default=0.5, help="Half-channel height H (mm).")
    ap.add_argument("--nx", type=int, default=4)
    ap.add_argument("--ny", type=int, default=240)
    ap.add_argument("--q", type=int, default=8, help="Quadrature order.")

    # Time stepping (days)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--t-final", type=float, default=7.0)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument(
        "--adaptive-dt",
        action="store_true",
        help="Enable adaptive time-step reduction on Newton failure (recommended for long runs).",
    )
    ap.add_argument("--dt-min", type=float, default=0.0, help="Minimum dt when using --adaptive-dt (0 disables).")
    ap.add_argument(
        "--dt-reduction-factor",
        type=float,
        default=0.5,
        help="dt <- factor*dt on failed step when using --adaptive-dt.",
    )

    # Alpha interface parameters (mm)
    ap.add_argument("--l0", type=float, default=0.10, help="Initial thickness (mm).")
    ap.add_argument("--eps-alpha", type=float, default=0.01, help="Diffuse interface half-thickness ε_α (mm).")
    ap.add_argument("--D-alpha", type=float, default=0.0, help="Alpha diffusion coefficient (mm^2/day).")
    ap.add_argument("--ac-M", type=float, default=1.0, help="Allen–Cahn mobility M_α (dimensionless). Use 0 to disable.")
    ap.add_argument("--ac-gamma", type=float, default=1.0, help="Allen–Cahn energy weight γ_α (dimensionless). Use 0 to disable.")

    # Prescribed skeleton velocity vS_y = g*y (mm/day)
    ap.add_argument("--growth-rate", type=float, default=0.2, help="Stretching rate g (1/day) for vS_y=g*y.")

    # Detachment law selection
    ap.add_argument("--model", type=str, default="shear", choices=("shear", "poly", "l2"))
    ap.add_argument("--poly-order", type=int, default=2, help="Truncation order for the polynomial approximation (model=poly).")

    # Duddu/Rittman parameters (defaults are the values used in main.tex for the 1D example)
    ap.add_argument("--mu", type=float, default=0.001, help="Dynamic viscosity mu (Pa*s).")
    ap.add_argument("--u-max0", type=float, default=12.5, help="Initial maximum velocity u_max^0 (mm/s).")
    ap.add_argument("--a", type=float, default=0.1, help="Detachment prefactor a (mm/day)/(Pa^b).")
    ap.add_argument("--b", type=float, default=0.5, help="Detachment exponent b.")
    ap.add_argument("--k-l2", type=float, default=0.28, help="Coefficient k for F_det=k*l^2 (mm/day).")

    # Solver parameters
    ap.add_argument("--newton-tol", type=float, default=1.0e-10)
    ap.add_argument("--max-it", type=int, default=40)
    ap.add_argument("--ls-mode", type=str, default="dealii")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    H = float(args.H)
    p = DudduRittmanParams(
        H_mm=float(args.H),
        mu_Pa_s=float(args.mu),
        u_max0_mm_s=float(args.u_max0),
        a_mm_per_day_per_sqrt_Pa=float(args.a),
        b=float(args.b),
    )

    # ---- mesh / spaces ----
    nodes, elems, _, corners = structured_quad(float(args.Lx), H, nx=int(args.nx), ny=int(args.ny), poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    _tag_rectangle_boundaries(mesh, L=float(args.Lx), H=H)

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
        "S": 1,
    }
    me = MixedElement(mesh, field_specs=field_specs)
    dh = DofHandler(me, method="cg")

    # Solve only alpha; treat all other fields as frozen coefficients.
    _mark_inactive_fields(dh, "v_x", "v_y", "p", "vS_x", "vS_y", "u_x", "u_y", "phi", "S")

    V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
    VS = FunctionSpace("VS", ["vS_x", "vS_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    dv = VectorTrialFunction(space=V, dof_handler=dh)
    dvS = VectorTrialFunction(space=VS, dof_handler=dh)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    dp = TrialFunction("p", dof_handler=dh)
    dphi = TrialFunction("phi", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    dS = TrialFunction("S", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    vS_test = VectorTestFunction(space=VS, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    S_test = TestFunction("S", dof_handler=dh)

    # Unknowns (k) and previous (n)
    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    vS_k = VectorFunction("vS_k", ["vS_x", "vS_y"], dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    S_k = Function("S_k", "S", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    vS_n = VectorFunction("vS_n", ["vS_x", "vS_y"], dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    S_n = Function("S_n", "S", dof_handler=dh)

    # Frozen fields
    for vf in (v_k, vS_k, u_k, v_n, vS_n, u_n):
        vf.nodal_values[:] = 0.0
    for sf in (p_k, p_n, S_k, S_n):
        sf.nodal_values[:] = 0.0
    for pf in (phi_k, phi_n):
        pf.nodal_values[:] = 0.3

    # Prescribed growth velocity vS_y = g*y (mm/day) and vS_x = 0.
    g = float(args.growth_rate)
    coords_vSy = np.asarray(dh.get_dof_coords("vS_y"), dtype=float)
    if coords_vSy.ndim == 2 and coords_vSy.shape[1] >= 2:
        y = coords_vSy[:, 1]
        vSy_vals = g * y
        sl = np.asarray(dh.get_field_slice("vS_y"), dtype=int).ravel()
        vS_k.set_nodal_values(sl, vSy_vals)
        vS_n.set_nodal_values(sl, vSy_vals)

    # Initial alpha profile
    def alpha0(x, y):
        # Use tanh((h-y)/(2*eps_alpha)) so that |∇alpha|≈δ(alpha)/(4*eps_alpha),
        # consistent with the scaling used in model.tex (and the D_det ↔ V_det mapping).
        return _alpha_step_eval(x, y, h=float(args.l0), eps=2.0 * float(args.eps_alpha))

    alpha_n.set_values_from_function(alpha0)
    alpha_k.nodal_values[:] = alpha_n.nodal_values[:]

    dt_c = Constant(float(args.dt))
    D_det_prev = Constant(0.0)

    # Build full one-domain forms, but solve only the alpha equation (r_alpha/a_alpha).
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
        dx=dx(metadata={"q": int(args.q)}),
        dt=dt_c,
        theta=float(args.theta),
        rho_f=Constant(1.0),
        mu_f=Constant(1.0),
        kappa_inv=Constant(1.0),
        mu_s=Constant(1.0),
        lambda_s=Constant(1.0),
        D_phi=0.0,
        gamma_phi=0.0,
        D_alpha=float(args.D_alpha),
        alpha_advection_form="advective",
        alpha_cahn_M=float(args.ac_M),
        alpha_cahn_gamma=float(args.ac_gamma),
        alpha_cahn_eps=float(args.eps_alpha),
        alpha_cahn_mobility="constant",
        alpha_cahn_conservative=False,
        D_S=0.0,
        mu_max=0.0,
        K_S=1.0,
        k_g=0.0,
        k_d=0.0,
        Y=1.0,
        rho_s_star=1.0,
        k_det=0.0,
        D_det_prev=D_det_prev,
        s_v=Constant(0.0),
        ds_v=Constant(0.0),
        gamma_u=0.0,
        u_extension_mode="l2",
        gamma_u_pin=0.0,
    )
    if forms.r_alpha is None or forms.a_alpha is None:
        raise RuntimeError("Expected per-block alpha forms (r_alpha/a_alpha) to be built.")

    # No BCs: we solve a pure Neumann/no-flux alpha equation on the strip.
    bcs: list[BoundaryCondition] = []
    bcs_homog: list[BoundaryCondition] = []

    lin_backend = str(args.linear_solver).strip().lower()
    if lin_backend == "petsc" and not HAS_PETSC:
        lin_backend = "scipy"

    rows: list[dict[str, float]] = []
    l_ode = float(_strip_thickness_alpha_half(dh=dh, alpha=alpha_n))
    t_ode = 0.0
    # Conversion between physical recession speed F_det [mm/day] and the rate coefficient
    # D_det_prev [1/day] multiplying δ(alpha)=4α(1-α).
    #
    # In model.tex, the heuristic scaling uses |∇α|≈δ/(4ε). For the *Allen–Cahn*
    # equilibrium profile with W'(α)=2α(1-α)(1-2α), the exact 1D interface profile is
    # α(s)=0.5(1+tanh(s/(√2 ε))) which gives |∇α| = δ/(2√2 ε) = √2 * δ/(4ε).
    #
    # Therefore, when Allen–Cahn regularization is enabled (ac_M*ac_gamma != 0),
    # we use an extra √2 factor so that the imposed speed matches the intended F_det.
    ac_enabled = (float(args.ac_M) != 0.0) and (float(args.ac_gamma) != 0.0)
    speed_to_rate_factor = math.sqrt(2.0) if ac_enabled else 1.0

    def _F_det(l_mm: float) -> float:
        m = str(args.model).strip().lower()
        if m == "shear":
            return detachment_speed_shear_mm_per_day(l_mm=float(l_mm), p=p)
        if m == "poly":
            return detachment_speed_poly_mm_per_day(l_mm=float(l_mm), p=p, order=int(args.poly_order))
        if m == "l2":
            return detachment_speed_l2_mm_per_day(l_mm=float(l_mm), k_l2=float(args.k_l2))
        raise ValueError(f"Unknown model {args.model!r}.")

    # Initialize D_det_prev from the initial thickness.
    F0 = _F_det(l_ode)
    eps = max(float(args.eps_alpha), 1.0e-12)
    D_det_prev.value = float(speed_to_rate_factor * F0 / (4.0 * eps))

    solver_ref: dict[str, object] = {}

    def _record_and_update(_funcs):
        solver_i = solver_ref.get("solver")
        if solver_i is None:
            return
        t_k = float(solver_i._current_t + solver_i._current_dt)
        alpha_k_i = alpha_k
        L_pde = float(_strip_thickness_alpha_half(dh=dh, alpha=alpha_k_i))

        nonlocal l_ode, t_ode
        dt_step = float(getattr(dt_c, "value", float(args.dt)))

        def rhs(_t, l):
            return float(g * l - _F_det(l))

        l_ode = float(_rk4_step(rhs, float(l_ode), float(t_ode), float(dt_step)))
        l_ode = float(max(l_ode, 0.0))
        t_ode = float(t_k)

        F_det_now = float(_F_det(L_pde))
        D_det_prev.value = float(speed_to_rate_factor * F_det_now / (4.0 * eps))

        rows.append(
            {
                "t_days": float(t_k),
                "L_pde_mm": float(L_pde),
                "L_ode_mm": float(l_ode),
                "F_det_mm_per_day": float(F_det_now),
                "D_det_prev_per_day": float(D_det_prev.value),
            }
        )

    def _on_dt_change(new_dt: float) -> None:
        dt_c.value = float(new_dt)

    solver = PdasNewtonSolver(
        forms.r_alpha,
        forms.a_alpha,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        vi_params=VIParameters(c=0.0, active_tol=0.0, project_initial_guess=True),
        newton_params=NewtonParameters(
            newton_tol=float(args.newton_tol),
            max_newton_iter=int(args.max_it),
            ls_mode=str(args.ls_mode),
            line_search=True,
        ),
        lin_params=LinearSolverParameters(backend=str(lin_backend)),
        quad_order=int(args.q),
        backend=str(args.backend),
        postproc_timeloop_cb=_record_and_update,
    )
    solver.set_box_bounds(by_field={"alpha": (0.0, 1.0)})
    solver_ref["solver"] = solver

    functions = [v_k, p_k, vS_k, u_k, phi_k, alpha_k, S_k]
    prev_functions = [v_n, p_n, vS_n, u_n, phi_n, alpha_n, S_n]

    solver.solve_time_interval(
        functions=functions,
        prev_functions=prev_functions,
        aux_functions={"dt": dt_c},
        time_params=TimeStepperParameters(
            dt=float(args.dt),
            final_time=float(args.t_final),
            max_steps=int(1.0e9),
            theta=float(args.theta),
            t0=0.0,
            stop_on_steady=False,
            on_dt_change=_on_dt_change,
            allow_dt_reduction=bool(getattr(args, "adaptive_dt", False)),
            dt_min=float(getattr(args, "dt_min", 0.0) or 0.0),
            dt_reduction_factor=float(getattr(args, "dt_reduction_factor", 0.5) or 0.5),
        ),
    )

    outpath = outdir / f"model={str(args.model)}_backend={str(args.backend)}_timeseries.csv"
    _write_csv(outpath, rows)
    print(f"[ok] wrote {outpath}")


if __name__ == "__main__":
    main()
