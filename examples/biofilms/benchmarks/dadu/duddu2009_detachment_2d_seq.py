"""
Duddu et al. (2009) detachment-model comparison in a 2D flow cell (one-domain model).

Reference
---------
R. Duddu, D. L. Chopp, B. Moran (2009)
"A Two-Dimensional Continuum Model of Biofilm Growth Incorporating Fluid Flow and Shear
 Stress Based Detachment", Biotechnol. Bioeng. 103:92–104.

Detachment models compared (paper: Fig. 6)
-----------------------------------------
The paper compares three continuous detachment-speed functions (interface recession speeds):
  (a) shear-based:    F_det = a |tau|^b,
  (b) height-based:   F_det = k_l2 * l^2,
  (c) polynomial:     F_det = k0*(1 + 2l + 4l^2)   (the b=1/2 1D expansion with H=0.5mm).

In the one-domain diffuse-interface implementation, detachment enters the alpha equation as
  + D_det_prev * δ(alpha)  in the residual,
which corresponds to a physical recession speed V_det via:
  D_det_prev ≈ (V_det / (4 eps_alpha)) * speed_to_rate_factor.

When Allen–Cahn regularization is enabled (ac_M*ac_gamma != 0), the equilibrium profile
for the specific W'(alpha) used in one_domain.py yields speed_to_rate_factor = sqrt(2).

Scope of this benchmark
-----------------------
This driver is intended to compare detachment *models* in 2D in a controlled setting.
To keep the setup robust and fast:
  - flow is solved as Stokes–Brinkman (set rho_f=0 -> no inertia/NS convection),
  - skeleton (u,vS) and porosity phi are frozen coefficients,
  - growth is driven by the implemented alpha "logistic" source G(S,phi)*alpha*(1-alpha),
    optionally coupled to a substrate advection–diffusion equation.

All files and outputs are kept inside examples/biofilms/benchmarks/dadu/.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.solvers.nonlinear_solver import (
    HAS_PETSC,
    LinearSolverParameters,
    NewtonParameters,
    NewtonSolver,
    PdasNewtonSolver,
    TimeStepperParameters,
    VIParameters,
)
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    grad,
    inner,
)
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad

from examples.utils.biofilm.adhesion import assemble_scalar
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


def _set_inactive_fields(dh: DofHandler, field_names: list[str]) -> None:
    inactive: set[int] = set()
    for fname in field_names:
        try:
            sl = np.asarray(dh.get_field_slice(fname), dtype=int).ravel()
        except Exception:
            continue
        inactive.update(int(i) for i in sl)
    dh.dof_tags = {"inactive": inactive}


def _as_float_time(fn):
    return lambda x, y, t: float(np.asarray(fn(np.asarray(x), np.asarray(y), float(t))).reshape(()))


def _smooth_step(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(np.asarray(z, dtype=float)))


def _alpha_semicircle_union(x, y, *, centers_x: list[float], r: float, eps: float) -> np.ndarray:
    """
    Union of semicircular colonies on the bottom wall (center at (x_i,0), radius r).
    Diffuse interface uses a tanh profile with half-thickness ~eps.
    """
    eps = max(float(eps), 1.0e-12)
    xx, yy = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    a = np.zeros_like(xx, dtype=float)
    for cx in centers_x:
        d = np.sqrt((xx - float(cx)) ** 2 + (yy - 0.0) ** 2)
        a_i = _smooth_step((float(r) - d) / (2.0 * eps))
        a = np.maximum(a, a_i)
    # clamp for safety
    return np.clip(a, 0.0, 1.0)


def _parabolic_inlet_u(y: np.ndarray, *, H: float, U_avg: float) -> np.ndarray:
    """
    Plane Poiseuille profile on y in [0,H] with average velocity U_avg:
        u(y) = 6 U_avg (y/H) (1 - y/H).
    """
    y = np.asarray(y, dtype=float)
    H = float(H)
    if H <= 0.0:
        return np.zeros_like(y)
    z = np.clip(y / H, 0.0, 1.0)
    return 6.0 * float(U_avg) * z * (1.0 - z)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    keys = list(rows[0].keys())

    def _py(val: object) -> object:
        try:
            # numpy scalar -> python scalar
            if isinstance(val, np.generic):
                return val.item()
        except Exception:
            pass
        return val

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _py(r.get(k)) for k in keys})


def _flush_timeseries_csv(*, path: Path, rows: list[dict[str, object]], every: int, step_no: int) -> None:
    """
    Periodically flush the full time series to CSV so long runs are inspectable
    and partial results aren't lost on late-step failures.
    """
    n = int(every or 0)
    if n <= 0:
        return
    if int(step_no) == 0 or (int(step_no) % n == 0):
        _write_csv(path, rows)


def _height_profile_alpha_half(
    *,
    dh: DofHandler,
    alpha: Function,
    alpha_half: float = 0.5,
    x_round: int = 10,
    y_round: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a coarse thickness profile l(x) from nodal alpha by:
      - grouping alpha nodes by x-level (rounded),
      - for each x-level, taking max y where alpha>=alpha_half.

    Returns (x_levels, l_levels) in mesh-coordinate units.
    """
    coords = np.asarray(dh.get_dof_coords("alpha"), dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        return np.asarray([]), np.asarray([])
    x = coords[:, 0]
    y = coords[:, 1]
    a = np.asarray(alpha.nodal_values, dtype=float).ravel()
    if x.size != a.size:
        return np.asarray([]), np.asarray([])

    # Round to structured-grid levels.
    x_key = np.round(x, decimals=int(x_round))
    y_key = np.round(y, decimals=int(y_round))
    x_levels, inv = np.unique(x_key, return_inverse=True)

    l = np.zeros_like(x_levels, dtype=float)
    for i in range(x_levels.size):
        mask = (inv == i) & np.isfinite(a) & (a >= float(alpha_half))
        if not np.any(mask):
            l[i] = 0.0
        else:
            l[i] = float(np.max(y_key[mask]))

    return np.asarray(x_levels, dtype=float), np.asarray(l, dtype=float)


def _l_profile_stats(l_prof: np.ndarray) -> tuple[float, float, float]:
    l = np.asarray(l_prof, dtype=float).ravel()
    if l.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.max(l)), float(np.mean(l)), float(np.std(l))


@dataclass(frozen=True)
class DetachmentModelSpec:
    key: str
    label: str


def _parse_models(raw: str) -> list[DetachmentModelSpec]:
    raw = str(raw or "").strip().lower()
    if raw in {"all", "*"}:
        raw = "shear,l2,poly"
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    out: list[DetachmentModelSpec] = []
    for p in parts:
        if p in {"none", "0", "off"}:
            out.append(DetachmentModelSpec(key="none", label="none"))
        elif p in {"shear", "tau"}:
            out.append(DetachmentModelSpec(key="shear", label="shear"))
        elif p in {"l2", "height", "xavier"}:
            out.append(DetachmentModelSpec(key="l2", label="l2"))
        elif p in {"poly", "polynomial"}:
            out.append(DetachmentModelSpec(key="poly", label="poly"))
        else:
            raise ValueError(f"Unknown detachment model '{p}'. Use 'shear', 'l2', 'poly', or 'none'.")
    if not out:
        raise ValueError("Empty --models list.")
    # de-duplicate but keep order
    seen = set()
    uniq: list[DetachmentModelSpec] = []
    for m in out:
        if m.key in seen:
            continue
        seen.add(m.key)
        uniq.append(m)
    return uniq


def _build_detachment_rate(
    *,
    model_key: str,
    mu_f: Constant,
    mu_tau: Constant,
    v_prev: VectorFunction,
    alpha_prev: Function,
    eps_alpha: float,
    ac_enabled: bool,
    shear_tau_model: str,
    normal_eta: float,
    # shear params (paper)
    shear_a: float,
    shear_b: float,
    shear_eta: float,
    # height params (paper)
    k_l2: float,
    k0_poly: float,
    H: float,
) -> object:
    """
    Return D_det_prev (rate coefficient, 1/day) to be used in the alpha equation.

    For height-based models we use Analytic(x,y)->rate(x,y).
    For shear-based we return a UFL expression in terms of v_prev.
    """
    eps = max(float(eps_alpha), 1.0e-12)
    speed_to_rate = math.sqrt(2.0) if bool(ac_enabled) else 1.0
    c_speed_to_rate = Constant(float(speed_to_rate))
    c_inv_4eps = Constant(1.0 / (4.0 * eps))

    key = str(model_key).strip().lower()
    if key == "none":
        return Constant(0.0)

    if key == "l2":
        # V_det(y) = k_l2 * y^2  (y is distance to substratum in this flat-bottom setup).
        k = float(k_l2)
        return c_speed_to_rate * Analytic(lambda x, y: (k * (np.asarray(y, dtype=float) ** 2)) * float(c_inv_4eps.value), degree=2)

    if key == "poly":
        # Paper's truncated polynomial for H=0.5mm: V_det = k0*(1 + 2l + 4l^2), l in mm.
        # For general H, keep the same formula but interpret l=y and coefficients as chosen by the user.
        k0 = float(k0_poly)
        return c_speed_to_rate * Analytic(
            lambda x, y: (k0 * (1.0 + (2.0 * np.asarray(y, dtype=float)) + (4.0 * (np.asarray(y, dtype=float) ** 2))))
            * float(c_inv_4eps.value),
            degree=2,
        )

    if key == "shear":
        # Detachment speed: V_det = a * |tau|^b.
        #
        # Two tau models:
        # - "proxy":  |tau| ≈ 2*mu_tau*||eps(v)||_F (robust, used elsewhere in one_domain.py)
        # - "paper":  tau as in Duddu (2009) Eq. (5), using n ~ grad(alpha)/||grad(alpha)||
        b = float(shear_b)
        if b <= 0.0:
            raise ValueError(f"shear_b must be > 0; got {b}.")
        tau_model = str(shear_tau_model or "").strip().lower()
        if tau_model in {"proxy", "eps", "strain"}:
            epsv = 0.5 * (grad(v_prev) + grad(v_prev).T)
            tau_proxy = Constant(2.0) * mu_tau * (inner(epsv, epsv) ** Constant(0.5))
            tau = tau_proxy
        elif tau_model in {"paper", "duddu", "interface"}:
            # pycutfem's `grad(...)` is scalar-only (returns a 2-vector). For a vector
            # field v=(u,v) we therefore build the needed derivatives componentwise.
            du_dx = grad(v_prev[0])[0]
            du_dy = grad(v_prev[0])[1]
            dv_dx = grad(v_prev[1])[0]
            dv_dy = grad(v_prev[1])[1]

            ga = grad(alpha_prev)
            ga2 = inner(ga, ga)
            denom = (ga2 + Constant(float(max(normal_eta, 0.0)))) ** Constant(0.5)
            nx = ga[0] / denom
            ny = ga[1] / denom

            term1 = (nx * nx - ny * ny) * (du_dy + dv_dx)
            term2 = Constant(2.0) * nx * ny * (dv_dy - du_dx)
            tau = mu_tau * (term1 + term2)
        else:
            raise ValueError(
                f"Unknown --shear-tau-model={shear_tau_model!r}. Use 'paper' or 'proxy'."
            )

        tau_abs = (tau * tau + Constant(float(max(shear_eta, 0.0)))) ** Constant(0.5)
        V = Constant(float(shear_a)) * (tau_abs ** Constant(b))
        return c_speed_to_rate * V * c_inv_4eps

    raise ValueError(f"Unknown detachment model {model_key!r}.")


def _run_one_model(
    *,
    model: DetachmentModelSpec,
    args,
) -> Path:
    outdir = Path(args.outdir) / f"model={model.label}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[run] model={model.label} backend={args.backend} nx={args.nx} ny={args.ny} dt={args.dt} t_final={args.t_final}")

    L = float(args.L)
    H = float(args.H)
    nx = int(args.nx)
    ny = int(args.ny)
    qdeg = int(args.q)

    # Mesh
    nodes, elems, _, corners = structured_quad(L, H, nx=nx, ny=ny, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    _tag_rectangle_boundaries(mesh, L=L, H=H)

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

    # Base inactive fields: freeze skeleton/solid/porosity (we only compare detachment laws here).
    base_inactive_fields = ["vS_x", "vS_y", "u_x", "u_y", "phi"]
    if not bool(args.solve_substrate):
        base_inactive_fields.append("S")
    _set_inactive_fields(dh, base_inactive_fields)

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
    for vf in (vS_k, u_k, vS_n, u_n):
        vf.nodal_values[:] = 0.0

    phi_b = float(args.phi_b)
    phi_n.set_values_from_function(lambda x, y: phi_b)
    phi_k.nodal_values[:] = phi_n.nodal_values[:]

    # Initial biofilm colonies
    r0 = float(args.r0)
    eps0 = float(args.eps0)
    ncol = int(args.n_colonies)
    if ncol <= 0:
        raise ValueError("--n-colonies must be >= 1")
    if r0 <= 0.0:
        raise ValueError("--r0 must be > 0")
    # Place colonies with uniform spacing, avoid touching boundaries.
    margin = max(2.0 * r0, 1.0e-12)
    if float(L) <= 2.0 * margin:
        raise ValueError("Domain too small for requested colonies/radius.")
    xs = np.linspace(margin, float(L) - margin, ncol).tolist()

    alpha_n.set_values_from_function(lambda x, y: _alpha_semicircle_union(x, y, centers_x=xs, r=r0, eps=eps0))
    alpha_k.nodal_values[:] = alpha_n.nodal_values[:]

    # Substrate initial/bulk
    S_in = float(args.S_in)
    S_n.set_values_from_function(lambda x, y: S_in)
    S_k.nodal_values[:] = S_n.nodal_values[:]

    # Initial guess: inlet profile everywhere (helps Newton).
    U_avg = float(args.U_avg)
    coords_vx = np.asarray(dh.get_dof_coords("v_x"), dtype=float)
    if coords_vx.ndim == 2 and coords_vx.shape[1] >= 2:
        y = coords_vx[:, 1]
        vx = _parabolic_inlet_u(y, H=H, U_avg=U_avg)
        sl_vx = np.asarray(dh.get_field_slice("v_x"), dtype=int).ravel()
        v_n.set_nodal_values(sl_vx, vx)
        v_k.set_nodal_values(sl_vx, vx)
    coords_vy = np.asarray(dh.get_dof_coords("v_y"), dtype=float)
    if coords_vy.ndim == 2 and coords_vy.shape[1] >= 2:
        sl_vy = np.asarray(dh.get_field_slice("v_y"), dtype=int).ravel()
        zeros = np.zeros(int(sl_vy.size), dtype=float)
        v_n.set_nodal_values(sl_vy, zeros)
        v_k.set_nodal_values(sl_vy, zeros)
    p_n.nodal_values[:] = 0.0
    p_k.nodal_values[:] = 0.0

    # Time / interface regularization
    dt_c = Constant(float(args.dt))
    eps_alpha = float(args.eps_alpha)
    ac_M = float(args.ac_M)
    ac_gamma = float(args.ac_gamma)
    ac_enabled = (ac_M != 0.0) and (ac_gamma != 0.0)

    mu_f = Constant(float(args.mu_f))
    mu_tau = mu_f
    if getattr(args, "mu_tau", None) is not None:
        mu_tau = Constant(float(args.mu_tau))
    D_det_prev = _build_detachment_rate(
        model_key=model.key,
        mu_f=mu_f,
        mu_tau=mu_tau,
        v_prev=v_n,
        alpha_prev=alpha_n,
        eps_alpha=eps_alpha,
        ac_enabled=ac_enabled,
        shear_tau_model=str(args.shear_tau_model),
        normal_eta=float(args.normal_eta),
        shear_a=float(args.shear_a),
        shear_b=float(args.shear_b),
        shear_eta=float(args.shear_eta),
        k_l2=float(args.k_l2),
        k0_poly=float(args.k0_poly),
        H=float(H),
    )

    # Build one-domain forms; only assemble the active blocks.
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
        dx=dx(metadata={"q": qdeg}),
        dt=dt_c,
        theta=float(args.theta),
        # Stokes–Brinkman: set rho_f=0 to remove inertia/convection.
        rho_f=Constant(0.0),
        mu_f=mu_f,
        kappa_inv=Constant(float(args.kappa_inv)),
        mu_s=Constant(0.0),
        lambda_s=Constant(0.0),
        solid_model="linear",
        mu_b_model=str(args.mu_b_model),
        # Porosity frozen; keep its diffusion/stabilization off.
        D_phi=0.0,
        gamma_phi=0.0,
        # Alpha regularization
        D_alpha=float(args.D_alpha),
        alpha_advection_form="advective",
        alpha_cahn_M=float(ac_M),
        alpha_cahn_gamma=float(ac_gamma),
        alpha_cahn_eps=float(eps_alpha),
        alpha_cahn_mobility="constant",
        alpha_cahn_conservative=False,
        # Substrate (optional)
        D_S=float(args.D_S),
        substrate_reaction_scheme=str(args.substrate_reaction_scheme),
        substrate_diffusion_scheme=str(args.substrate_diffusion_scheme),
        mu_max=float(args.mu_max),
        K_S=float(args.K_S),
        k_g=float(args.k_g),
        k_d=float(args.k_d),
        Y=float(args.Y),
        rho_s_star=float(args.rho_s_star),
        # Surface detachment coefficient
        D_det_prev=D_det_prev,
        # No volume source (keep div(C v)=0)
        s_v=Constant(0.0),
        ds_v=Constant(0.0),
        # No u extension (u frozen)
        gamma_u=0.0,
        u_extension_mode="l2",
        gamma_u_pin=0.0,
    )

    # Assemble only needed blocks (drop skeleton/phi/kinematics equations).
    residual_form = forms.r_momentum + forms.r_mass + forms.r_alpha
    jacobian_form = forms.a_momentum + forms.a_mass + forms.a_alpha
    if bool(args.solve_substrate):
        residual_form = residual_form + forms.r_substrate
        jacobian_form = jacobian_form + forms.a_substrate

    # BCs (time-independent)
    bcs: list[BoundaryCondition] = []

    bcs.append(
        BoundaryCondition(
            "v_x",
            "dirichlet",
            "left",
            _as_float_time(lambda x, y, t: _parabolic_inlet_u(y, H=H, U_avg=U_avg)),
        )
    )
    bcs.append(BoundaryCondition("v_y", "dirichlet", "left", _as_float_time(lambda x, y, t: 0.0)))
    for tag in ("top", "bottom"):
        bcs.append(BoundaryCondition("v_x", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)))
        bcs.append(BoundaryCondition("v_y", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)))

    # Pressure gauge at the outlet
    bcs.append(BoundaryCondition("p", "dirichlet", "right", _as_float_time(lambda x, y, t: 0.0)))

    if bool(args.solve_substrate):
        bcs.append(BoundaryCondition("S", "dirichlet", "left", _as_float_time(lambda x, y, t: S_in)))

    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, (lambda x, y: 0.0)) for b in bcs]

    # Time series
    dx_q = dx(metadata={"q": qdeg})
    rows: list[dict[str, float]] = []
    outpath = outdir / f"backend={str(args.backend)}_timeseries.csv"
    flush_every = int(getattr(args, "flush_csv_every", 0) or 0)

    # Area window metric (for paper-like comparisons that exclude the inlet colony).
    x0 = args.area_x0
    x1 = args.area_x1
    y0 = args.area_y0
    y1 = args.area_y1
    if all(v is not None for v in (x0, x1, y0, y1)):
        x0f, x1f, y0f, y1f = float(x0), float(x1), float(y0), float(y1)

        def _chi_window(x, y):
            xx = np.asarray(x, dtype=float)
            yy = np.asarray(y, dtype=float)
            return ((x0f <= xx) & (xx <= x1f) & (y0f <= yy) & (yy <= y1f)).astype(float)

        chi_window = Analytic(_chi_window, degree=0)
    else:
        chi_window = None

    # Optional VTK output
    vtk_every = int(args.vtk_every)
    if vtk_every < 0:
        vtk_every = -1

    solver_ref: dict[str, object] = {}

    def _append_row(*, t_k: float, step_no: int) -> None:
        # Metrics
        A_alpha = float(assemble_scalar(dh, alpha_k * dx_q, backend=str(args.backend), quad_order=qdeg))
        if chi_window is None:
            A_alpha_window = float(A_alpha)
        else:
            A_alpha_window = float(
                assemble_scalar(dh, alpha_k * chi_window * dx_q, backend=str(args.backend), quad_order=qdeg)
            )
        x_levels, l_prof = _height_profile_alpha_half(dh=dh, alpha=alpha_k, alpha_half=float(args.alpha_half))
        L_max, L_mean, L_std = _l_profile_stats(l_prof)

        S_min = float(np.min(S_k.nodal_values)) if bool(args.solve_substrate) else float(S_in)
        S_max = float(np.max(S_k.nodal_values)) if bool(args.solve_substrate) else float(S_in)

        rows.append(
            {
                "t_days": float(t_k),
                "A_alpha": float(A_alpha),
                "A_alpha_window": float(A_alpha_window),
                "L_max": float(L_max),
                "L_mean": float(L_mean),
                "L_std": float(L_std),
                "S_min": float(S_min),
                "S_max": float(S_max),
            }
        )
        _flush_timeseries_csv(path=outpath, rows=rows, every=flush_every, step_no=int(step_no))

        if vtk_every > 0 and ((step_no > 0 and step_no % vtk_every == 0) or abs(t_k - float(args.t_final)) < 1.0e-12):
            export_vtk(
                filename=str(outdir / f"solution_{step_no:05d}.vtu"),
                mesh=mesh,
                dof_handler=dh,
                functions={
                    "v": v_k,
                    "p": p_k,
                    "alpha": alpha_k,
                    "S": S_k if bool(args.solve_substrate) else (lambda x, y: float(S_in)),
                },
            )

    def _post_cb(_funcs):
        solver_i = solver_ref.get("solver")
        if solver_i is None:
            return
        t_k = float(solver_i._current_t + solver_i._current_dt)
        step_no = int(getattr(solver_i, "_current_step_no", len(rows) + 1))
        _append_row(t_k=t_k, step_no=step_no)

    def _on_dt_change(new_dt: float) -> None:
        dt_c.value = float(new_dt)

    functions = [v_k, p_k, vS_k, u_k, phi_k, alpha_k, S_k]
    prev_functions = [v_n, p_n, vS_n, u_n, phi_n, alpha_n, S_n]

    # Always record the initial state (also makes --t-final 0.0 robust).
    _append_row(t_k=0.0, step_no=0)

    lin_backend = str(args.linear_solver).strip().lower()
    if lin_backend == "petsc" and not HAS_PETSC:
        lin_backend = "scipy"

    strategy = str(getattr(args, "strategy", "sequential")).strip().lower()
    if strategy not in {"monolithic", "split", "sequential"}:
        raise ValueError("--strategy must be 'monolithic', 'split', or 'sequential'.")

    t0 = time.perf_counter()
    if strategy == "monolithic":
        _set_inactive_fields(dh, base_inactive_fields)
        solver = PdasNewtonSolver(
            residual_form,
            jacobian_form,
            dof_handler=dh,
            mixed_element=me,
            bcs=bcs,
            bcs_homog=bcs_homog,
            vi_params=VIParameters(
                c=float(args.vi_c),
                active_tol=float(args.vi_active_tol),
                project_initial_guess=True,
                project_each_iteration=True,
            ),
            newton_params=NewtonParameters(
                newton_tol=float(args.newton_tol),
                newton_rtol=float(getattr(args, "newton_rtol", 0.0) or 0.0),
                max_newton_iter=int(args.max_it),
                print_level=int(args.print_level),
                ls_mode=str(args.ls_mode),
                line_search=bool(args.line_search),
            ),
            lin_params=LinearSolverParameters(backend=str(lin_backend)),
            quad_order=qdeg,
            backend=str(args.backend),
            postproc_timeloop_cb=_post_cb,
        )
        bounds = {"alpha": (0.0, 1.0)}
        if bool(args.solve_substrate) and bool(getattr(args, "S_bounds", True)):
            bounds["S"] = (0.0, None)
        solver.set_box_bounds(by_field=bounds)
        solver_ref["solver"] = solver

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
    elif strategy == "split":
        # Split strategy:
        #   (1) solve Stokes–Brinkman (v,p) with alpha,S frozen,
        #   (2) solve (alpha,S) with v,p frozen.
        flow_inactive_fields = base_inactive_fields + ["alpha"]
        if bool(args.solve_substrate):
            flow_inactive_fields.append("S")
        growth_inactive_fields = base_inactive_fields + ["v_x", "v_y", "p"]
        if not bool(args.solve_substrate):
            growth_inactive_fields.append("S")

        # Flow solver (unconstrained).
        _set_inactive_fields(dh, flow_inactive_fields)
        solver_flow = NewtonSolver(
            forms.r_momentum + forms.r_mass,
            forms.a_momentum + forms.a_mass,
            dof_handler=dh,
            mixed_element=me,
            bcs=bcs,
            bcs_homog=bcs_homog,
            newton_params=NewtonParameters(
                # The flow block is linear here (rho_f=0, coefficients frozen), so a
                # very tight absolute tolerance can lead to stagnation at roundoff.
                newton_tol=float(max(float(args.newton_tol), 1.0e-8)),
                newton_rtol=float(getattr(args, "newton_rtol", 0.0) or 0.0),
                max_newton_iter=int(args.max_it),
                print_level=int(args.print_level),
                ls_mode=str(args.ls_mode),
                line_search=bool(args.line_search) and bool(getattr(args, "flow_line_search", False)),
            ),
            lin_params=LinearSolverParameters(backend=str(lin_backend)),
            quad_order=qdeg,
            backend=str(args.backend),
        )

        # Growth/substrate solver (bounded).
        _set_inactive_fields(dh, growth_inactive_fields)
        res_g = forms.r_alpha
        jac_g = forms.a_alpha
        if bool(args.solve_substrate):
            res_g = res_g + forms.r_substrate
            jac_g = jac_g + forms.a_substrate
        solver_g = PdasNewtonSolver(
            res_g,
            jac_g,
            dof_handler=dh,
            mixed_element=me,
            bcs=bcs,
            bcs_homog=bcs_homog,
            vi_params=VIParameters(
                c=float(args.vi_c),
                active_tol=float(args.vi_active_tol),
                project_initial_guess=True,
                project_each_iteration=True,
            ),
            newton_params=NewtonParameters(
                newton_tol=float(args.newton_tol),
                newton_rtol=float(getattr(args, "newton_rtol", 0.0) or 0.0),
                max_newton_iter=int(args.max_it),
                print_level=int(args.print_level),
                ls_mode=str(args.ls_mode),
                line_search=bool(args.line_search),
            ),
            lin_params=LinearSolverParameters(backend=str(lin_backend)),
            quad_order=qdeg,
            backend=str(args.backend),
        )
        bounds = {"alpha": (0.0, 1.0)}
        if bool(args.solve_substrate) and bool(getattr(args, "S_bounds", True)):
            bounds["S"] = (0.0, None)
        solver_g.set_box_bounds(by_field=bounds)

        t_n = 0.0
        t_final = float(args.t_final)
        dt_nom = float(args.dt)
        dt_min = float(getattr(args, "dt_min", 0.0) or 0.0)
        red = float(getattr(args, "dt_reduction_factor", 0.5) or 0.5)
        step_no = 0
        tol_t = 1.0e-12

        while t_n < t_final - tol_t:
            step_no += 1
            dt_step = min(dt_nom, t_final - t_n)

            while True:
                if dt_min > 0.0 and dt_step < dt_min - 0.0:
                    raise RuntimeError(f"Split strategy: dt={dt_step:.3e} dropped below dt_min={dt_min:.3e}.")

                dt_c.value = float(dt_step)

                # Predictor (k <- n)
                for f, f_prev in zip(functions, prev_functions):
                    f.nodal_values[:] = f_prev.nodal_values[:]

                # Apply time-dependent BCs at t_{n+theta}.
                t_bc = float(t_n + float(args.theta) * dt_step)
                bcs_now = solver_flow._freeze_bcs(bcs, t_bc)
                dh.apply_bcs(bcs_now, *functions)

                try:
                    # (1) Flow solve with alpha,S frozen.
                    solver_flow._current_step_no = int(step_no)
                    solver_flow._current_t = float(t_n)
                    solver_flow._current_dt = float(dt_step)
                    solver_flow._newton_loop(functions, prev_functions, {"dt": dt_c}, bcs_now)

                    # Promote flow fields so detachment uses the current flow.
                    v_n.nodal_values[:] = v_k.nodal_values[:]
                    p_n.nodal_values[:] = p_k.nodal_values[:]

                    # (2) Growth/substrate solve with v,p frozen.
                    solver_g._current_step_no = int(step_no)
                    solver_g._current_t = float(t_n)
                    solver_g._current_dt = float(dt_step)
                    solver_g._newton_loop(functions, prev_functions, {"dt": dt_c}, bcs_now)

                except Exception as exc:
                    if not bool(getattr(args, "adaptive_dt", False)):
                        raise
                    dt_step *= float(red)
                    msg = str(exc).strip()
                    tag = f"{type(exc).__name__}: {msg}" if msg else f"{type(exc).__name__}"
                    print(f"    Rejecting step {step_no}; reducing Δt → {dt_step:.3e} ({tag}) and retrying.")
                    continue

                # Accept: promote k -> n for all fields.
                for f_prev, f in zip(prev_functions, functions):
                    f_prev.nodal_values[:] = f.nodal_values[:]

                t_n += float(dt_step)
                _append_row(t_k=float(t_n), step_no=int(step_no))
                break
    else:
        # Sequential strategy (paper-like operator splitting):
        #   (1) solve Stokes–Brinkman (v,p) with alpha,S frozen,
        #   (2) solve substrate S with v,p,alpha frozen,
        #   (3) solve alpha with v,p,S frozen.
        flow_inactive_fields = base_inactive_fields + ["alpha"]
        if bool(args.solve_substrate):
            flow_inactive_fields.append("S")

        # Substrate solve: keep everything except S inactive.
        substrate_inactive_fields = base_inactive_fields + ["v_x", "v_y", "p", "alpha"]
        # Alpha solve: keep everything except alpha inactive.
        alpha_inactive_fields = base_inactive_fields + ["v_x", "v_y", "p", "S"]

        _set_inactive_fields(dh, flow_inactive_fields)
        solver_flow = NewtonSolver(
            forms.r_momentum + forms.r_mass,
            forms.a_momentum + forms.a_mass,
            dof_handler=dh,
            mixed_element=me,
            bcs=bcs,
            bcs_homog=bcs_homog,
            newton_params=NewtonParameters(
                # The flow block is linear here (rho_f=0, coefficients frozen), so a
                # very tight absolute tolerance can lead to stagnation at roundoff.
                newton_tol=float(max(float(args.newton_tol), 1.0e-8)),
                newton_rtol=float(getattr(args, "newton_rtol", 0.0) or 0.0),
                max_newton_iter=int(args.max_it),
                print_level=int(args.print_level),
                ls_mode=str(args.ls_mode),
                line_search=bool(args.line_search) and bool(getattr(args, "flow_line_search", False)),
            ),
            lin_params=LinearSolverParameters(backend=str(lin_backend)),
            quad_order=qdeg,
            backend=str(args.backend),
        )

        solver_sub = None
        if bool(args.solve_substrate):
            _set_inactive_fields(dh, substrate_inactive_fields)
            solver_sub = NewtonSolver(
                forms.r_substrate,
                forms.a_substrate,
                dof_handler=dh,
                mixed_element=me,
                bcs=bcs,
                bcs_homog=bcs_homog,
                newton_params=NewtonParameters(
                    newton_tol=float(args.newton_tol),
                    newton_rtol=float(getattr(args, "newton_rtol", 0.0) or 0.0),
                    max_newton_iter=int(args.max_it),
                    print_level=int(args.print_level),
                    ls_mode=str(args.ls_mode),
                    line_search=bool(args.line_search),
                ),
                lin_params=LinearSolverParameters(backend=str(lin_backend)),
                quad_order=qdeg,
                backend=str(args.backend),
            )

        _set_inactive_fields(dh, alpha_inactive_fields)
        solver_alpha = PdasNewtonSolver(
            forms.r_alpha,
            forms.a_alpha,
            dof_handler=dh,
            mixed_element=me,
            bcs=bcs,
            bcs_homog=bcs_homog,
            vi_params=VIParameters(
                c=float(args.vi_c),
                active_tol=float(args.vi_active_tol),
                project_initial_guess=True,
                project_each_iteration=True,
            ),
            newton_params=NewtonParameters(
                newton_tol=float(args.newton_tol),
                newton_rtol=float(getattr(args, "newton_rtol", 0.0) or 0.0),
                max_newton_iter=int(args.max_it),
                print_level=int(args.print_level),
                ls_mode=str(args.ls_mode),
                line_search=bool(args.line_search),
            ),
            lin_params=LinearSolverParameters(backend=str(lin_backend)),
            quad_order=qdeg,
            backend=str(args.backend),
        )
        solver_alpha.set_box_bounds(by_field={"alpha": (0.0, 1.0)})

        t_n = 0.0
        t_final = float(args.t_final)
        dt_nom = float(args.dt)
        dt_min = float(getattr(args, "dt_min", 0.0) or 0.0)
        red = float(getattr(args, "dt_reduction_factor", 0.5) or 0.5)
        step_no = 0
        tol_t = 1.0e-12

        while t_n < t_final - tol_t:
            step_no += 1
            dt_step = min(dt_nom, t_final - t_n)

            while True:
                if dt_min > 0.0 and dt_step < dt_min - 0.0:
                    raise RuntimeError(f"Sequential strategy: dt={dt_step:.3e} dropped below dt_min={dt_min:.3e}.")

                dt_c.value = float(dt_step)

                # Predictor (k <- n)
                for f, f_prev in zip(functions, prev_functions):
                    f.nodal_values[:] = f_prev.nodal_values[:]

                # Apply time-dependent BCs at t_{n+theta}.
                t_bc = float(t_n + float(args.theta) * dt_step)
                bcs_now = solver_flow._freeze_bcs(bcs, t_bc)
                dh.apply_bcs(bcs_now, *functions)

                try:
                    # (1) Flow solve with alpha,S frozen.
                    solver_flow._current_step_no = int(step_no)
                    solver_flow._current_t = float(t_n)
                    solver_flow._current_dt = float(dt_step)
                    solver_flow._newton_loop(functions, prev_functions, {"dt": dt_c}, bcs_now)

                    # Promote flow fields so detachment and substrate advection use the current flow.
                    v_n.nodal_values[:] = v_k.nodal_values[:]
                    p_n.nodal_values[:] = p_k.nodal_values[:]

                    # (2) Substrate solve with v,p,alpha frozen.
                    if solver_sub is not None:
                        solver_sub._current_step_no = int(step_no)
                        solver_sub._current_t = float(t_n)
                        solver_sub._current_dt = float(dt_step)
                        solver_sub._newton_loop(functions, prev_functions, {"dt": dt_c}, bcs_now)

                        # Keep substrate non-negative to avoid Monod singularities.
                        S_k.nodal_values[:] = np.maximum(np.asarray(S_k.nodal_values, dtype=float), 0.0)
                        S_n.nodal_values[:] = S_k.nodal_values[:]

                    # (3) Alpha solve with v,p,S frozen.
                    solver_alpha._current_step_no = int(step_no)
                    solver_alpha._current_t = float(t_n)
                    solver_alpha._current_dt = float(dt_step)
                    solver_alpha._newton_loop(functions, prev_functions, {"dt": dt_c}, bcs_now)

                except Exception as exc:
                    if not bool(getattr(args, "adaptive_dt", False)):
                        raise
                    dt_step *= float(red)
                    msg = str(exc).strip()
                    tag = f"{type(exc).__name__}: {msg}" if msg else f"{type(exc).__name__}"
                    print(f"    Rejecting step {step_no}; reducing Δt → {dt_step:.3e} ({tag}) and retrying.")
                    continue

                # Accept: promote k -> n for all fields.
                for f_prev, f in zip(prev_functions, functions):
                    f_prev.nodal_values[:] = f.nodal_values[:]

                t_n += float(dt_step)
                _append_row(t_k=float(t_n), step_no=int(step_no))
                break

    t_solve = time.perf_counter() - t0

    _write_csv(outpath, rows)
    print(f"[ok] wrote {outpath} (solve {t_solve:.2f}s)")
    return outpath


def main() -> None:
    ap = argparse.ArgumentParser(description="Duddu2009 2D detachment-model comparison (one-domain, reduced blocks).")
    ap.add_argument("--models", type=str, default="shear,l2,poly", help="Comma list: shear,l2,poly,none or 'all'.")
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--linear-solver", type=str, default=("petsc" if HAS_PETSC else "scipy"), choices=("petsc", "scipy"))
    ap.add_argument("--outdir", type=str, default="examples/biofilms/benchmarks/dadu/results/duddu2009_detachment_2d")

    # Geometry / mesh (mm)
    ap.add_argument("--L", type=float, default=2.0)
    ap.add_argument("--H", type=float, default=0.5)
    ap.add_argument("--nx", type=int, default=120)
    ap.add_argument("--ny", type=int, default=30)
    ap.add_argument("--q", type=int, default=4)

    # Inlet flow (mm/day). (Use small values for robustness; this driver uses Stokes by default.)
    ap.add_argument("--U-avg", type=float, default=10.0, help="Average inlet velocity (mm/day).")
    ap.add_argument(
        "--U-avg-mm-s",
        type=float,
        default=None,
        help="Average inlet velocity (mm/s); converted to mm/day and overrides --U-avg.",
    )

    # Initial colonies (mm)
    ap.add_argument("--n-colonies", type=int, default=5)
    ap.add_argument("--r0", type=float, default=0.05, help="Initial semicircle radius (mm).")
    ap.add_argument("--eps0", type=float, default=0.01, help="Initial interface thickness for alpha0 (mm).")

    # Time stepping (days)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--t-final", type=float, default=1.0)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--strategy", type=str, default="sequential", choices=("monolithic", "split", "sequential"))
    ap.add_argument(
        "--adaptive-dt",
        action="store_true",
        help="Enable adaptive time-step reduction on Newton failure (requires on_dt_change, already set).",
    )
    ap.add_argument("--dt-min", type=float, default=0.0, help="Minimum dt when using --adaptive-dt (0 disables).")
    ap.add_argument(
        "--dt-reduction-factor",
        type=float,
        default=0.5,
        help="dt <- factor*dt on failed step when using --adaptive-dt.",
    )

    # One-domain coefficients / regularization
    ap.add_argument("--phi-b", type=float, default=0.3, help="Frozen biofilm porosity phi (inside alpha≈1).")
    ap.add_argument(
        "--mu-f",
        type=float,
        default=1.0,
        help="Dynamic viscosity used in Stokes–Brinkman (also scales Brinkman drag); choose a numerically convenient scaling.",
    )
    ap.add_argument(
        "--mu-f-Pa-s",
        type=float,
        default=None,
        help="Dynamic viscosity mu_f in Pa*s; converted to Pa*day (divide by 86400) and overrides --mu-f.",
    )
    ap.add_argument(
        "--mu-tau",
        type=float,
        default=None,
        help="Dynamic viscosity used only for shear-stress in detachment (defaults to --mu-f).",
    )
    ap.add_argument(
        "--mu-tau-Pa-s",
        type=float,
        default=None,
        help="Dynamic viscosity used only for detachment shear-stress in Pa*s; converted to Pa*day (divide by 86400) and overrides --mu-tau/--mu-f.",
    )
    ap.add_argument("--mu-b-model", type=str, default="mu", choices=("mu", "phi_mu"))
    ap.add_argument("--kappa-inv", type=float, default=1.0e3, help="Inverse permeability (bigger -> more solid-like biofilm).")

    # Alpha regularization (mm)
    ap.add_argument("--eps-alpha", type=float, default=0.01, help="Interface thickness eps_alpha used in D_det mapping (mm).")
    ap.add_argument("--D-alpha", type=float, default=0.0, help="Alpha diffusion (mm^2/day). Typically 0 when using Allen–Cahn.")
    ap.add_argument("--ac-M", type=float, default=1.0)
    ap.add_argument("--ac-gamma", type=float, default=1.0)
    ap.add_argument("--alpha-half", type=float, default=0.5, help="Alpha threshold used for height metrics.")

    # Growth / substrate (dimensionless-to-user scaling)
    ap.add_argument("--k-g", type=float, default=1.0, help="Growth prefactor in G(S,phi).")
    ap.add_argument("--mu-max", type=float, default=0.5, help="Monod maximum rate (1/day).")
    ap.add_argument("--K-S", type=float, default=0.1)
    ap.add_argument("--k-d", type=float, default=0.0)
    ap.add_argument("--Y", type=float, default=1.0)
    ap.add_argument("--rho-s-star", type=float, default=1.0)

    ap.add_argument("--solve-substrate", action="store_true", help="Also solve the substrate advection–diffusion equation.")
    ap.add_argument("--S-in", type=float, default=1.0, help="Inlet substrate value (used when solve-substrate is on; otherwise S is frozen).")
    ap.add_argument("--D-S", type=float, default=1.0e-3, help="Substrate diffusion (mm^2/day).")
    ap.add_argument("--substrate-reaction-scheme", type=str, default="implicit", choices=("theta", "implicit", "explicit"))
    ap.add_argument("--substrate-diffusion-scheme", type=str, default="implicit", choices=("theta", "implicit", "explicit"))
    ap.add_argument(
        "--no-S-bounds",
        dest="S_bounds",
        action="store_false",
        default=True,
        help="Disable the VI bound S>=0 (useful for convergence debugging; monitor S_min in the CSV).",
    )

    # Detachment: shear-based (paper form) and height-based (paper constants)
    ap.add_argument("--shear-a", type=float, default=0.1, help="Shear detachment prefactor a in V=a|tau|^b (speed units).")
    ap.add_argument("--shear-b", type=float, default=0.5, help="Shear detachment exponent b.")
    ap.add_argument(
        "--shear-tau-model",
        type=str,
        default="paper",
        choices=("paper", "proxy"),
        help="How to compute tau for shear detachment: Duddu (2009) Eq.(5) ('paper') or robust proxy ('proxy').",
    )
    ap.add_argument("--normal-eta", type=float, default=1.0e-12, help="Regularization used in n=grad(alpha)/sqrt(|grad(alpha)|^2+eta).")
    ap.add_argument("--shear-eta", type=float, default=1.0e-12, help="Regularization added inside sqrt for |tau| (units: Pa^2).")
    ap.add_argument("--k-l2", type=float, default=0.28, help="Height detachment V=k*l^2 (mm/day).")
    ap.add_argument("--k0-poly", type=float, default=0.00707, help="Polynomial detachment base coefficient (mm/day).")

    # Metrics
    ap.add_argument("--area-x0", type=float, default=None, help="Area metric window x0 (mm). Requires all 4 bounds.")
    ap.add_argument("--area-x1", type=float, default=None, help="Area metric window x1 (mm). Requires all 4 bounds.")
    ap.add_argument("--area-y0", type=float, default=None, help="Area metric window y0 (mm). Requires all 4 bounds.")
    ap.add_argument("--area-y1", type=float, default=None, help="Area metric window y1 (mm). Requires all 4 bounds.")

    # Output
    ap.add_argument("--vtk-every", type=int, default=0, help="Write VTK every N steps (0 disables).")
    ap.add_argument(
        "--flush-csv-every",
        type=int,
        default=0,
        help="Rewrite the time series CSV every N accepted steps (0 only writes at the end).",
    )

    # Solver params
    ap.add_argument("--newton-tol", type=float, default=1.0e-10)
    ap.add_argument("--newton-rtol", type=float, default=0.0)
    ap.add_argument("--max-it", type=int, default=40)
    ap.add_argument(
        "--print-level",
        type=int,
        default=1,
        choices=(0, 1, 2, 3),
        help="Newton/VI solver verbosity (0=quiet, 1=step summary, 2=iterations, 3=line-search/assembly details).",
    )
    ap.add_argument("--ls-mode", type=str, default="dealii")
    ap.add_argument(
        "--no-line-search",
        dest="line_search",
        action="store_false",
        default=True,
        help="Disable line-search backtracking and always take the full Newton step (sometimes needed for saddle-point blocks).",
    )
    ap.add_argument(
        "--flow-line-search",
        action="store_true",
        default=False,
        help="Enable line search for the flow (v,p) solve. Default is off because the Stokes–Brinkman block is linear in this benchmark.",
    )
    ap.add_argument("--vi-c", type=float, default=0.0)
    ap.add_argument("--vi-active-tol", type=float, default=0.0)

    args = ap.parse_args()

    # Convenience overrides for paper-like units while keeping "day" as the internal time unit.
    seconds_per_day = 86400.0
    if getattr(args, "U_avg_mm_s", None) is not None:
        args.U_avg = float(args.U_avg_mm_s) * seconds_per_day
    if getattr(args, "mu_f_Pa_s", None) is not None:
        # Use "day" as the internal time unit (v in mm/day), so convert
        # mu [Pa*s] -> mu [Pa*day] by dividing by seconds_per_day.
        args.mu_f = float(args.mu_f_Pa_s) / seconds_per_day
    if getattr(args, "mu_tau_Pa_s", None) is not None:
        args.mu_tau = float(args.mu_tau_Pa_s) / seconds_per_day

    logging.getLogger("pycutfem").setLevel(logging.WARNING)

    models = _parse_models(args.models)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, object]] = []
    for m in models:
        path = _run_one_model(model=m, args=args)
        # Record final row for quick comparison (if file exists and has rows).
        try:
            data = np.genfromtxt(str(path), delimiter=",", names=True, dtype=float, encoding=None)
            if getattr(data, "shape", ()) == ():
                data = np.array([data], dtype=data.dtype)
            last = data[-1]
            names = tuple(getattr(data.dtype, "names", ()) or ())
            summary.append(
                {
                    "model": str(m.label),
                    "t_final": float(last["t_days"]),
                    "A_alpha": float(last["A_alpha"]),
                    "A_alpha_window": float(last["A_alpha_window"]) if "A_alpha_window" in names else float(last["A_alpha"]),
                    "L_max": float(last["L_max"]),
                    "L_mean": float(last["L_mean"]),
                    "L_std": float(last["L_std"]) if "L_std" in names else 0.0,
                }
            )
        except Exception:
            pass

    # Optional summary CSV (numeric model index; compare by folder names).
    if summary:
        sum_path = outdir / "summary.csv"
        _write_csv(sum_path, summary)
        print(f"[ok] wrote {sum_path}")


if __name__ == "__main__":
    main()
