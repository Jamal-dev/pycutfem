"""
Zhang–Cogan–Wang (2008) cavity-growth benchmark for the one-domain biofilm model.

Paper (in this repo)
--------------------
`examples/biofilms/benchmarks/zhang/main.tex`

Aligned numerical example
-------------------------
Section 5.1 "Growth of biofilms in a cavity": a bottom-attached biofilm layer with a
single hump, no-slip walls, and a top-feeding substrate boundary condition.

What this driver does
---------------------
This script sets up the above geometry/BCs and runs the monolithic one-domain model
implemented in `examples/utils/biofilm/one_domain.py` with parameters chosen to
approximate the *extended Newtonian* (non-poroelastic) regime by:
  - disabling elastic stiffness (mu_s=lambda_s=0, and freezing `u`),
  - enabling an (optional) viscous skeleton response via `solid_visco_eta`,
  - using strong Brinkman drag to keep v ≈ vS when desired.

It writes a small time-series CSV with thickness-like diagnostics and field extrema,
intended for reproducible comparisons and for the companion investigation notes.

Run (recommended)
-----------------
  conda run --no-capture-output -n fenicsx \\
    python examples/biofilms/benchmarks/zhang/zhang2008_cavity_one_domain.py --backend cpp
"""

from __future__ import annotations

import argparse
import csv
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
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
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
    return 0.5 * (1.0 + np.tanh(np.asarray(z, dtype=float)))

def _one_minus(expr):
    # Keep the UFL operand on the left for backend compatibility (Constant - Function is fragile).
    return (-expr) + Constant(1.0)


def _alpha_step_eval(x: np.ndarray, y: np.ndarray, *, h: np.ndarray, eps: float) -> np.ndarray:
    """Diffuse Heaviside: alpha≈1 for y<h, alpha≈0 for y>h."""
    eps = max(float(eps), 1.0e-12)
    xx, yy = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    hh = np.broadcast_to(np.asarray(h, dtype=float), xx.shape)
    return _smooth_step((hh - yy) / eps)


def _single_hump_height(x: np.ndarray, *, Lx: float, h_base: float, hump_amp: float, hump_sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x0 = 0.5 * float(Lx)
    sig = max(float(hump_sigma), 1.0e-12)
    return float(h_base) + float(hump_amp) * np.exp(-0.5 * ((x - x0) / sig) ** 2)


def _as_float_time(fn):
    return lambda x, y, t: float(np.asarray(fn(np.asarray(x), np.asarray(y), float(t))).reshape(()))


@dataclass(frozen=True)
class StepDiagnostics:
    t: float
    L_eff: float
    H_max: float
    H_mean: float
    alpha_min: float
    alpha_max: float
    S_min: float
    S_max: float
    v_max: float
    p_max: float
    vS_max: float


def _thickness_diagnostics(
    *,
    dh: DofHandler,
    alpha: Function,
    Lx: float,
    backend: str,
    qdeg: int,
    alpha_half: float = 0.5,
    x_round: int = 10,
) -> tuple[float, float, float]:
    """Return (L_eff, H_max, H_mean) in nondimensional units."""
    dx_q = dx(metadata={"q": int(qdeg)})
    I_alpha = assemble_scalar(dh, alpha * dx_q, backend=backend, quad_order=int(qdeg))
    L_eff = float(I_alpha) / max(float(Lx), 1.0e-16)

    coords = np.asarray(dh.get_dof_coords("alpha"), dtype=float)
    vals = np.asarray(alpha.nodal_values, dtype=float).reshape(-1)
    if coords.ndim != 2 or coords.shape[0] != vals.shape[0]:
        return L_eff, float("nan"), float("nan")

    x = coords[:, 0]
    y = coords[:, 1]
    keys = np.round(x, int(x_round))
    uniq = np.unique(keys)
    heights = []
    for k in uniq:
        m = keys == k
        mm = m & (vals >= float(alpha_half))
        if not np.any(mm):
            continue
        heights.append(float(np.max(y[mm])))
    if not heights:
        return L_eff, 0.0, 0.0
    H_max = float(np.max(heights))
    H_mean = float(np.mean(heights))
    return L_eff, H_max, H_mean


def _build_problem(args) -> dict:
    Lx = float(args.Lx)
    Hy = float(args.Hy)
    nx = int(args.nx)
    ny = int(args.ny)
    qdeg = int(args.q)

    nodes, elems, _, corners = structured_quad(Lx, Hy, nx=nx, ny=ny, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    _tag_rectangle_boundaries(mesh, L=Lx, H=Hy)

    field_specs = {
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

    if bool(args.freeze_phi):
        _mark_inactive_fields(dh, "phi")
    if bool(args.freeze_u):
        _mark_inactive_fields(dh, "u_x", "u_y")

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

    for vf in (v_k, vS_k, u_k, v_n, vS_n, u_n):
        vf.nodal_values[:] = 0.0
    for sf in (p_k, p_n):
        sf.nodal_values[:] = 0.0

    def alpha0(x, y):
        h = _single_hump_height(x, Lx=Lx, h_base=float(args.h_base), hump_amp=float(args.hump_amp), hump_sigma=float(args.hump_sigma))
        return _alpha_step_eval(x, y, h=h, eps=float(args.eps))

    alpha_n.set_values_from_function(alpha0)
    alpha_k.nodal_values[:] = alpha_n.nodal_values[:]

    phi_b = float(args.phi_b)
    phi_n.set_values_from_function(lambda x, y: float(phi_b))
    phi_k.nodal_values[:] = phi_n.nodal_values[:]

    S0 = float(args.S0)
    S_n.set_values_from_function(lambda x, y: float(S0))
    S_k.nodal_values[:] = S_n.nodal_values[:]

    dt_c = Constant(float(args.dt))
    dx_q = dx(metadata={"q": int(qdeg)})

    # Volume source in the mixture constraint: div(C v + B vS) = alpha*s_v.
    s_v_key = str(args.s_v_mode).strip().lower()
    jac_key = str(args.s_v_jacobian).strip().lower()
    if jac_key in {"frozen", "lagged", "picard"}:
        jac_key = "frozen"
    elif jac_key in {"full", "newton", "consistent"}:
        jac_key = "full"
    else:
        raise ValueError(f"Unknown --s-v-jacobian={args.s_v_jacobian!r}. Use 'full' or 'frozen'.")

    mu_max_c = Constant(float(args.mu_max))
    K_S_c = Constant(float(args.K_S))
    k_d_c = Constant(float(args.k_d))

    if s_v_key in {"none", "0", "zero"}:
        s_v = Constant(0.0)
        ds_v = None
    elif s_v_key in {"mu", "mu_net", "munet", "rate"}:
        # Specific net growth rate (matches sharp-interface thickness growth when v≈vS).
        monod = mu_max_c * (S_k / (S_k + K_S_c))
        s_v = monod - k_d_c
        if jac_key == "full":
            denom = S_k + K_S_c
            dmonod = mu_max_c * (K_S_c / (denom * denom)) * dS
            ds_v = dmonod
        else:
            ds_v = None
    elif s_v_key in {"pi", "pib", "growth"}:
        # Biomass-weighted rate (closer to Zhang's g_n ∝ φ_n).
        monod = mu_max_c * (S_k / (S_k + K_S_c))
        s_v = (monod - k_d_c) * _one_minus(phi_k)
        if jac_key == "full":
            denom = S_k + K_S_c
            dmonod = mu_max_c * (K_S_c / (denom * denom)) * dS
            ds_v = dmonod * _one_minus(phi_k) - (monod - k_d_c) * dphi
        else:
            ds_v = None
    else:
        raise ValueError(f"Unknown --s-v-mode={args.s_v_mode!r}. Use 'none', 'mu', or 'pi'.")

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
        dx=dx_q,
        dt=dt_c,
        theta=float(args.theta),
        rho_f=Constant(float(args.rho_f)),
        mu_f=Constant(float(args.mu_f)),
        mu_b_model=str(args.mu_b_model),
        kappa_inv=Constant(float(args.kappa_inv)),
        solid_model="linear",
        mu_s=Constant(float(args.mu_s)),
        lambda_s=Constant(float(args.lambda_s)),
        solid_visco_eta=float(args.solid_visco_eta),
        D_phi=float(args.D_phi),
        gamma_phi=float(args.gamma_phi),
        D_alpha=float(args.D_alpha),
        alpha_advection_form=str(args.alpha_advection_form),
        alpha_cahn_M=float(args.alpha_cahn_M),
        alpha_cahn_gamma=float(args.alpha_cahn_gamma),
        alpha_cahn_eps=float(args.alpha_cahn_eps),
        alpha_cahn_conservative=bool(args.alpha_cahn_conservative),
        alpha_cahn_conservative_mode="eliminate",
        D_S=float(args.D_S),
        substrate_reaction_scheme=str(args.substrate_reaction_scheme),
        substrate_diffusion_scheme=str(args.substrate_diffusion_scheme),
        mu_max=float(args.mu_max),
        K_S=float(args.K_S),
        k_g=float(args.k_g),
        k_d=float(args.k_d),
        Y=float(args.Y),
        rho_s_star=float(args.rho_s_star),
        k_det=0.0,
        s_v=s_v,
        ds_v=ds_v,
        D_det_prev=Constant(0.0),
    )

    # Boundary conditions (cavity):
    # - no-slip for mixture velocity v at all boundaries,
    # - anchor skeleton (vS=0 and u=0) at all boundaries for robustness,
    # - fix pressure gauge (p=0) on the top boundary,
    # - top-feeding substrate: S= S_top at y=Hy.
    bcs: list[BoundaryCondition] = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                BoundaryCondition("v_x", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)),
                BoundaryCondition("v_y", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)),
                BoundaryCondition("vS_x", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)),
                BoundaryCondition("vS_y", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)),
                BoundaryCondition("u_x", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)),
                BoundaryCondition("u_y", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)),
            ]
        )
    bcs.append(BoundaryCondition("p", "dirichlet", "top", _as_float_time(lambda x, y, t: 0.0)))
    bcs.append(BoundaryCondition("S", "dirichlet", "top", _as_float_time(lambda x, y, t: float(args.S_top))))

    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, (lambda x, y: 0.0)) for b in bcs]

    fields = {"v": v_k, "p": p_k, "vS": vS_k, "u": u_k, "phi": phi_k, "alpha": alpha_k, "S": S_k}
    prev_fields = {"v": v_n, "p": p_n, "vS": vS_n, "u": u_n, "phi": phi_n, "alpha": alpha_n, "S": S_n}

    functions = [v_k, p_k, vS_k, u_k, phi_k, alpha_k, S_k]
    prev_functions = [v_n, p_n, vS_n, u_n, phi_n, alpha_n, S_n]

    return {
        "mesh": mesh,
        "me": me,
        "dh": dh,
        "dt_c": dt_c,
        "forms": forms,
        "bcs": bcs,
        "bcs_homog": bcs_homog,
        "fields": fields,
        "prev_fields": prev_fields,
        "functions": functions,
        "prev_functions": prev_functions,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Zhang (2008) cavity-growth benchmark using the one-domain biofilm model.")
    ap.add_argument("--outdir", type=str, default="out/zhang2008_cavity_one_domain")
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="Write .npz field snapshots every N time steps (0 disables).",
    )
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=32)
    ap.add_argument("--q", type=int, default=8, help="Quadrature order (dx metadata + solver quad_order).")
    ap.add_argument("--Lx", type=float, default=1.0)
    ap.add_argument("--Hy", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--t-final", type=float, default=50.0)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--newton-tol", type=float, default=1.0e-10)
    ap.add_argument("--newton-rtol", type=float, default=0.0)
    ap.add_argument("--max-it", type=int, default=30)
    ap.add_argument("--no-pdas", action="store_true", help="Use plain Newton (no VI bounds).")
    ap.add_argument("--vi-c", type=float, default=0.0, help="PDAS complementarity scaling (0 uses solver default).")

    # Initial hump: y = h_base + hump_amp * exp(-(x-L/2)^2/(2*hump_sigma^2))
    ap.add_argument("--h-base", type=float, default=0.2)
    ap.add_argument("--hump-amp", type=float, default=0.08)
    ap.add_argument("--hump-sigma", type=float, default=0.12)
    ap.add_argument("--eps", type=float, default=0.02, help="Diffuse interface thickness for the alpha IC.")

    # Substrate: top feeding + initial condition.
    ap.add_argument("--S-top", type=float, default=1.0)
    ap.add_argument("--S0", type=float, default=0.0)

    # Mixture parameters (dimensionless by default to match Zhang's nondimensionalization).
    ap.add_argument("--rho-f", type=float, default=1.0)
    ap.add_argument("--mu-f", type=float, default=1.0e0)
    ap.add_argument("--mu-b-model", type=str, default="mu", choices=("mu", "phi_mu"))
    ap.add_argument("--kappa-inv", type=float, default=1.0e6, help="Inverse permeability scaling for drag.")

    # "Non-poroelastic" toggle: set elastic moduli to 0 and freeze u by default.
    ap.add_argument("--freeze-u", type=int, default=1)
    ap.add_argument("--mu-s", type=float, default=0.0)
    ap.add_argument("--lambda-s", type=float, default=0.0)
    ap.add_argument("--solid-visco-eta", type=float, default=0.0, help="Optional Kelvin–Voigt viscosity on vS.")

    # Porosity/phi handling
    ap.add_argument("--phi-b", type=float, default=0.91, help="Constant biofilm porosity (phi) inside alpha≈1.")
    ap.add_argument("--freeze-phi", type=int, default=1)
    ap.add_argument("--D-phi", type=float, default=0.0)
    ap.add_argument("--gamma-phi", type=float, default=0.0)

    # Alpha transport + regularization
    ap.add_argument("--alpha-advection-form", type=str, default="advective", choices=("advective", "conservative"))
    ap.add_argument("--D-alpha", type=float, default=0.0)
    ap.add_argument("--alpha-cahn-M", type=float, default=0.0)
    ap.add_argument("--alpha-cahn-gamma", type=float, default=0.0)
    ap.add_argument("--alpha-cahn-eps", type=float, default=1.0)
    ap.add_argument("--alpha-cahn-conservative", type=int, default=0)

    # Substrate transport + kinetics
    ap.add_argument("--D-S", type=float, default=2.3)
    ap.add_argument(
        "--substrate-reaction-scheme",
        type=str,
        default="implicit",
        choices=("theta", "implicit", "imex", "explicit"),
    )
    ap.add_argument(
        "--substrate-diffusion-scheme",
        type=str,
        default="theta",
        choices=("theta", "implicit", "imex", "explicit"),
    )
    ap.add_argument("--mu-max", type=float, default=0.14)
    ap.add_argument("--K-S", type=float, default=0.15)
    ap.add_argument("--k-d", type=float, default=0.0)
    ap.add_argument("--Y", type=float, default=1.0)
    ap.add_argument("--rho-s-star", type=float, default=1.0)

    # Growth options
    ap.add_argument("--k-g", type=float, default=0.0, help="Interface-growth mapping coefficient (usually 0 for this benchmark).")
    ap.add_argument("--s-v-mode", type=str, default="mu", choices=("none", "mu", "pi"))
    ap.add_argument("--s-v-jacobian", type=str, default="full", choices=("full", "frozen"))

    args = ap.parse_args()
    outdir = Path(str(args.outdir)).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    snap_every = int(getattr(args, "snapshot_every", 0) or 0)
    snapdir = outdir / "snapshots"
    if snap_every > 0:
        snapdir.mkdir(parents=True, exist_ok=True)

    prob = _build_problem(args)
    dh: DofHandler = prob["dh"]

    backend = str(args.backend)
    qdeg = int(args.q)

    diag_rows: list[StepDiagnostics] = []

    def _record_step(_funcs):
        solver_i = solver_ref.get("solver")
        if solver_i is None:
            return
        step_no = int(getattr(solver_i, "_current_step_no", len(diag_rows) + 1))
        t_k = float(solver_i._current_t + solver_i._current_dt)
        fields = prob["fields"]
        alpha_k = fields["alpha"]
        S_k = fields["S"]
        v_k = fields["v"]
        p_k = fields["p"]
        vS_k = fields["vS"]
        L_eff, H_max, H_mean = _thickness_diagnostics(
            dh=dh, alpha=alpha_k, Lx=float(args.Lx), backend=backend, qdeg=qdeg, alpha_half=0.5
        )
        diag_rows.append(
            StepDiagnostics(
                t=t_k,
                L_eff=float(L_eff),
                H_max=float(H_max),
                H_mean=float(H_mean),
                alpha_min=float(np.min(alpha_k.nodal_values)),
                alpha_max=float(np.max(alpha_k.nodal_values)),
                S_min=float(np.min(S_k.nodal_values)),
                S_max=float(np.max(S_k.nodal_values)),
                v_max=float(np.max(np.abs(v_k.nodal_values))),
                p_max=float(np.max(np.abs(p_k.nodal_values))),
                vS_max=float(np.max(np.abs(vS_k.nodal_values))),
            )
        )

        if snap_every > 0 and (step_no % snap_every) == 0:
            data = {
                "t": np.asarray([t_k], dtype=float),
                "alpha_xy": np.asarray(dh.get_dof_coords("alpha"), dtype=float),
                "alpha": np.asarray(alpha_k.nodal_values, dtype=float),
                "S_xy": np.asarray(dh.get_dof_coords("S"), dtype=float),
                "S": np.asarray(S_k.nodal_values, dtype=float),
                "phi_xy": np.asarray(dh.get_dof_coords("phi"), dtype=float),
                "phi": np.asarray(fields["phi"].nodal_values, dtype=float),
                "p_xy": np.asarray(dh.get_dof_coords("p"), dtype=float),
                "p": np.asarray(p_k.nodal_values, dtype=float),
                "v_x_xy": np.asarray(dh.get_dof_coords("v_x"), dtype=float),
                "v_x": np.asarray(v_k.nodal_values[:, 0], dtype=float),
                "v_y_xy": np.asarray(dh.get_dof_coords("v_y"), dtype=float),
                "v_y": np.asarray(v_k.nodal_values[:, 1], dtype=float),
                "vS_x_xy": np.asarray(dh.get_dof_coords("vS_x"), dtype=float),
                "vS_x": np.asarray(vS_k.nodal_values[:, 0], dtype=float),
                "vS_y_xy": np.asarray(dh.get_dof_coords("vS_y"), dtype=float),
                "vS_y": np.asarray(vS_k.nodal_values[:, 1], dtype=float),
            }
            np.savez_compressed(snapdir / f"step{step_no:06d}_t={t_k:.6e}.npz", **data)

    solver_ref: dict[str, object] = {}

    newton_params = NewtonParameters(
        newton_tol=float(args.newton_tol),
        newton_rtol=float(args.newton_rtol),
        max_newton_iter=int(args.max_it),
        ls_mode="armijo",
    )
    newton_params.line_search = True

    use_pdas = not bool(args.no_pdas)
    if use_pdas:
        solver = PdasNewtonSolver(
            prob["forms"].residual_form,
            prob["forms"].jacobian_form,
            dof_handler=dh,
            mixed_element=prob["me"],
            bcs=prob["bcs"],
            bcs_homog=prob["bcs_homog"],
            vi_params=VIParameters(c=float(args.vi_c or 0.0), active_tol=0.0, project_initial_guess=True),
            newton_params=newton_params,
            quad_order=int(qdeg),
            backend=backend,
            postproc_timeloop_cb=_record_step,
        )
        solver.set_box_bounds(by_field={"alpha": (0.0, 1.0), "S": (0.0, None), "phi": (0.0, 1.0)})
    else:
        solver = NewtonSolver(
            prob["forms"].residual_form,
            prob["forms"].jacobian_form,
            dof_handler=dh,
            mixed_element=prob["me"],
            bcs=prob["bcs"],
            bcs_homog=prob["bcs_homog"],
            newton_params=newton_params,
            quad_order=int(qdeg),
            backend=backend,
            postproc_timeloop_cb=_record_step,
        )
    solver_ref["solver"] = solver

    def _on_dt_change(new_dt: float) -> None:
        prob["dt_c"].value = float(new_dt)

    solver.solve_time_interval(
        functions=prob["functions"],
        prev_functions=prob["prev_functions"],
        aux_functions={"dt": prob["dt_c"]},
        time_params=TimeStepperParameters(
            dt=float(args.dt),
            final_time=float(args.t_final),
            max_steps=int(1.0e9),
            theta=float(args.theta),
            t0=0.0,
            stop_on_steady=False,
            on_dt_change=_on_dt_change,
        ),
    )

    csv_path = outdir / f"timeseries_backend={backend}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "t",
                "L_eff",
                "H_max",
                "H_mean",
                "alpha_min",
                "alpha_max",
                "S_min",
                "S_max",
                "v_max",
                "p_max",
                "vS_max",
            ]
        )
        for r in diag_rows:
            w.writerow(
                [
                    f"{r.t:.16e}",
                    f"{r.L_eff:.16e}",
                    f"{r.H_max:.16e}",
                    f"{r.H_mean:.16e}",
                    f"{r.alpha_min:.16e}",
                    f"{r.alpha_max:.16e}",
                    f"{r.S_min:.16e}",
                    f"{r.S_max:.16e}",
                    f"{r.v_max:.16e}",
                    f"{r.p_max:.16e}",
                    f"{r.vS_max:.16e}",
                ]
            )

    print(f"[done] wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
