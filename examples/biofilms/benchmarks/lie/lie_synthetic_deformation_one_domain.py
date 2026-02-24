"""
Li et al. (2008) synthetic biofilm deformation benchmark (one-domain model).

Reference (LaTeX copy in this folder)
------------------------------------
Mengfei Li, Karel Matouš, Robert Nerenberg,
"Predicting biofilm deformation with a viscoelastic phase-field model:
 Modeling and experimental studies", available online 27 Feb 2008.

This driver aligns (as closely as possible within the current pycutfem one-domain
Navier–Stokes–Brinkman–Biot model) with the paper's *synthetic biofilm* channel-flow
deformation experiment:
  - channel height H = 10 mm, rho = 1000 kg/m^3, mu = 1e-3 Pa·s,
  - average inflow speed u_avg = 6e-4 m/s -> Re ≈ rho*u_avg*H/mu ≈ 6,
  - velocity ramp over Tramp = 1 s,
  - simulate to T = 20 s.

Key modeling choices vs the paper
---------------------------------
The paper uses a two-fluid phase-field + Oldroyd-B viscoelastic constitutive law.
The one-domain model in `examples/utils/biofilm/one_domain.py` uses a poroelastic
skeleton coupled to a single mixture velocity via Brinkman drag.

To produce a comparable "deforming biofilm blob" test while keeping the run robust
and focused on deformation, this driver:
  - disables growth/detachment/chemistry,
  - transports the indicator alpha via the Eulerian reference map:
        alpha(x,t) = alpha0(x - u(x,t)),
    i.e. we do not solve the alpha PDE in this benchmark,
  - ties porosity to alpha via phi = 1 - (1-phi_b)*alpha,
  - maps (G_b, mu_b) from the paper to a Kelvin–Voigt skeleton:
        mu_s = G_b,  eta_s = mu_b,
    with a near-incompressible lambda_s computed from nu.

Outputs
-------
Writes to `out_dir`:
  - `timeseries.csv`: tracking-line displacements (alpha=0.5 contour) vs time,
  - `vtk/step=XXXX.vtu` (optional): VTK snapshots for visualization.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

import numpy as np

# Avoid extremely verbose Numba debug dumps if the environment enables them.
# These are read at import time by Numba, so set them before importing pycutfem.
for _k in (
    "NUMBA_DEBUG",
    "NUMBA_DUMP_BYTECODE",
    "NUMBA_DUMP_IR",
    "NUMBA_DUMP_SSA",
    "NUMBA_DEBUG_ARRAY_OPT",
):
    os.environ[_k] = "0"

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.solvers.nonlinear_solver import (
    NewtonParameters,
    PetscSnesNewtonSolver,
    TimeStepperParameters,
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
    # Robust sigmoid: 0.5*(1+tanh(z)).
    return 0.5 * (1.0 + np.tanh(np.asarray(z, dtype=float)))


def _alpha_block_eval(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x0: float,
    w: float,
    h: float,
    eps: float,
) -> np.ndarray:
    """
    Smooth indicator for a bottom-attached rectangle:
      alpha ≈ 1 in [x0,x0+w]×[0,h], alpha ≈ 0 outside.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    eps = max(float(eps), 1.0e-12)
    sx0 = _smooth_step((x - float(x0)) / eps)
    sx1 = _smooth_step((float(x0 + w) - x) / eps)
    sy = _smooth_step((float(h) - y) / eps)
    return sx0 * sx1 * sy


def _lame_lambda_from_nu(mu: float, nu: float) -> float:
    nu = float(nu)
    if not (-1.0 < nu < 0.5):
        raise ValueError(f"nu must satisfy -1 < nu < 0.5; got nu={nu}")
    return float(2.0 * float(mu) * nu / max(1.0e-16, (1.0 - 2.0 * nu)))


def _build_coord_lookup(coords: np.ndarray, *, ndigits: int = 12) -> dict[tuple[float, float], int]:
    out: dict[tuple[float, float], int] = {}
    for i, xy in enumerate(np.asarray(coords, dtype=float)):
        out[(round(float(xy[0]), ndigits), round(float(xy[1]), ndigits))] = int(i)
    return out


def _nearest_y_levels(y_all: np.ndarray, targets: list[float]) -> list[float]:
    y = np.unique(np.asarray(y_all, dtype=float))
    if y.size == 0:
        return [float(t) for t in targets]
    out = []
    for t in targets:
        j = int(np.argmin(np.abs(y - float(t))))
        out.append(float(y[j]))
    return out


def _x_alpha_half_on_y_line(alpha_xy: np.ndarray, alpha_vals: np.ndarray, *, y_line: float, alpha_half: float = 0.5) -> float:
    xy = np.asarray(alpha_xy, dtype=float)
    a = np.asarray(alpha_vals, dtype=float).ravel()
    if xy.ndim != 2 or xy.shape[1] < 2 or a.shape[0] != xy.shape[0]:
        return float("nan")

    # Structured meshes give exact y-levels; use exact match with a tiny tol.
    mask = np.abs(xy[:, 1] - float(y_line)) <= 1.0e-14
    if not np.any(mask):
        return float("nan")

    x = np.asarray(xy[mask, 0], dtype=float).ravel()
    av = np.asarray(a[mask], dtype=float).ravel()
    if x.size < 2:
        return float("nan")

    order = np.argsort(x)
    x = x[order]
    av = av[order]

    # Find the last index where alpha is still above the threshold, then
    # interpolate with the next node (assumes a single right-edge crossing).
    above = av >= float(alpha_half)
    if not np.any(above):
        return float("nan")
    i = int(np.max(np.nonzero(above)[0]))
    if i >= x.size - 1:
        return float(x[-1])
    a0 = float(av[i])
    a1 = float(av[i + 1])
    x0 = float(x[i])
    x1 = float(x[i + 1])
    da = a1 - a0
    if abs(da) <= 1.0e-16:
        return float(x0)
    t = (float(alpha_half) - a0) / da
    return float(x0 + t * (x1 - x0))


def main() -> None:
    ap = argparse.ArgumentParser(description="Li et al. (2008) synthetic biofilm deformation benchmark (one-domain).")
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--nx", type=int, default=200)
    ap.add_argument("--ny", type=int, default=80)
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--t-final", type=float, default=20.0)
    ap.add_argument("--theta", type=float, default=1.0, help="Theta-scheme (1=BE, 0.5=CN).")
    ap.add_argument("--newton-tol", type=float, default=1.0e-9)
    ap.add_argument("--max-it", type=int, default=25)
    ap.add_argument("--out-dir", type=str, default="out/lie_synthetic_one_domain")
    ap.add_argument("--vtk-every", type=int, default=10, help="Write VTK every N steps (0 disables).")

    # Geometry + interface
    ap.add_argument("--L", type=float, default=25.0e-3)
    ap.add_argument("--H", type=float, default=10.0e-3)
    ap.add_argument("--bio-x0", type=float, default=10.0e-3, help="Biofilm block start x0 [m].")
    ap.add_argument("--bio-w", type=float, default=1.0e-3, help="Biofilm block width [m].")
    ap.add_argument("--bio-h", type=float, default=0.8e-3, help="Biofilm block height [m].")
    ap.add_argument("--eps", type=float, default=2.0e-4, help="Diffuse interface half-thickness used in alpha0 [m].")
    ap.add_argument("--phi-b", type=float, default=0.47, help="Porosity inside biofilm (phi_b).")

    # Flow: paper provides average u1; we convert to a parabolic profile.
    ap.add_argument("--u-avg", type=float, default=6.0e-4, help="Average inflow speed u_avg [m/s].")
    ap.add_argument("--t-ramp", type=float, default=1.0, help="Linear ramp time for inflow [s].")

    # Material parameters (paper Table 1 for synthetic biofilm)
    ap.add_argument("--rho-f", type=float, default=1000.0)
    ap.add_argument("--mu-f", type=float, default=1.0e-3)
    ap.add_argument("--G-b", type=float, default=69736.0, help="Shear modulus (mapped to mu_s) [Pa].")
    ap.add_argument("--mu-b", type=float, default=30494.0, help="Viscosity (mapped to solid_visco_eta) [Pa*s].")
    ap.add_argument("--nu", type=float, default=0.49, help="Poisson ratio for near-incompressible linear elastic skeleton.")

    # Poroelastic coupling
    ap.add_argument("--kappa-inv", type=float, default=1.0e12, help="Inverse permeability kappa^{-1} [1/m^2].")
    ap.add_argument("--gamma-u", type=float, default=5.0, help="u extension penalty outside biofilm.")
    ap.add_argument(
        "--u-extension",
        type=str,
        default="l2",
        choices=("l2", "grad"),
        help="Extension mode for u outside biofilm (matches build_biofilm_one_domain_forms).",
    )
    ap.add_argument(
        "--gamma-u-pin",
        type=float,
        default=1.0e-4,
        help="Tiny L2 pinning used only with --u-extension grad (removes translation nullspace).",
    )
    ap.add_argument("--gamma-phi", type=float, default=5.0, help="Penalty enforcing phi->1 in free fluid.")
    args = ap.parse_args()

    logging.getLogger("pycutfem.ufl.compilers").setLevel(logging.WARNING)

    backend = str(args.backend)
    L = float(args.L)
    H = float(args.H)
    dt_val = float(args.dt)
    theta = float(args.theta)
    qdeg = int(args.q)

    u_avg = float(args.u_avg)
    Umax = 1.5 * u_avg  # plane Poiseuille: u_max = 1.5 u_avg
    t_ramp = max(1.0e-12, float(args.t_ramp))

    # ------------------------------------------------------------------
    # Mesh + tags
    # ------------------------------------------------------------------
    nodes, elems, _, corners = structured_quad(L, H, nx=int(args.nx), ny=int(args.ny), poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    _tag_rectangle_boundaries(mesh, L=L, H=H)

    # ------------------------------------------------------------------
    # Mixed space
    # ------------------------------------------------------------------
    me = MixedElement(
        mesh,
        field_specs={
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
        },
    )
    dh = DofHandler(me, method="cg")

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

    # Unknowns (k) and previous state (n)
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

    # ------------------------------------------------------------------
    # Initial conditions (t=0)
    # ------------------------------------------------------------------
    v_n.nodal_values[:] = 0.0
    p_n.nodal_values[:] = 0.0
    vS_n.nodal_values[:] = 0.0
    u_n.nodal_values[:] = 0.0
    S_n.nodal_values[:] = 1.0

    alpha_xy = np.asarray(dh.get_dof_coords("alpha"), dtype=float)
    alpha0 = _alpha_block_eval(
        alpha_xy[:, 0],
        alpha_xy[:, 1],
        x0=float(args.bio_x0),
        w=float(args.bio_w),
        h=float(args.bio_h),
        eps=float(args.eps),
    )
    alpha_n.nodal_values[:] = np.asarray(alpha0, dtype=float)
    phi_b = float(args.phi_b)
    phi_n.nodal_values[:] = 1.0 - (1.0 - phi_b) * np.asarray(alpha0, dtype=float)

    # Freeze transport fields: alpha/phi/S are updated (alpha from refmap, phi tied to alpha) but not solved.
    _mark_inactive_fields(dh, "alpha", "phi", "S")

    # ------------------------------------------------------------------
    # Alpha-from-refmap mapping: alpha(x,t) = alpha0(x - u(x,t))
    # Build alpha->u DOF lookup via coordinate matching (Q1 alpha is a subset of Q2 u).
    # ------------------------------------------------------------------
    ux_xy = np.asarray(dh.get_dof_coords("u_x"), dtype=float)
    ux_lut = _build_coord_lookup(ux_xy, ndigits=12)
    alpha_to_u = np.full(alpha_xy.shape[0], -1, dtype=int)
    for i, xy in enumerate(alpha_xy):
        key = (round(float(xy[0]), 12), round(float(xy[1]), 12))
        j = ux_lut.get(key, None)
        if j is not None:
            alpha_to_u[i] = int(j)
    if np.any(alpha_to_u < 0):
        missing = int(np.sum(alpha_to_u < 0))
        raise RuntimeError(f"Failed mapping {missing} alpha DOFs to u DOFs; cannot use refmap alpha transport.")

    def _update_alpha_phi_from_refmap() -> None:
        ux = np.asarray(u_k.components[0].nodal_values, dtype=float).ravel()
        uy = np.asarray(u_k.components[1].nodal_values, dtype=float).ravel()
        u_at_alpha = np.column_stack([ux[alpha_to_u], uy[alpha_to_u]])
        chi = alpha_xy - u_at_alpha
        a = _alpha_block_eval(
            chi[:, 0],
            chi[:, 1],
            x0=float(args.bio_x0),
            w=float(args.bio_w),
            h=float(args.bio_h),
            eps=float(args.eps),
        )
        alpha_k.nodal_values[:] = np.clip(np.asarray(a, dtype=float), 0.0, 1.0)
        phi_k.nodal_values[:] = np.clip(1.0 - (1.0 - phi_b) * np.asarray(alpha_k.nodal_values, dtype=float), 0.0, 1.0)
        # Keep frozen substrate non-negative.
        S_k.nodal_values[:] = np.maximum(np.asarray(S_k.nodal_values, dtype=float), 0.0)

    # ------------------------------------------------------------------
    # Tracking lines (match the paper's "lines 1-3" concept)
    # ------------------------------------------------------------------
    hb = float(args.bio_h)
    y_targets = [0.25 * hb, 0.5 * hb, 0.75 * hb]
    a0_vals = np.asarray(alpha_n.nodal_values, dtype=float).ravel()
    bio_mask0 = a0_vals >= 0.5
    y_src = alpha_xy[bio_mask0, 1] if np.any(bio_mask0) else alpha_xy[:, 1]
    y_lines = _nearest_y_levels(y_src, y_targets)
    x_ref = [
        _x_alpha_half_on_y_line(alpha_xy, a0_vals, y_line=y0) for y0 in y_lines
    ]

    # ------------------------------------------------------------------
    # Forms
    # ------------------------------------------------------------------
    mu_s = float(args.G_b)
    lam_s = _lame_lambda_from_nu(mu_s, float(args.nu))
    rho_f = float(args.rho_f)
    mu_f = float(args.mu_f)

    # Paper check (informational): Re=rho*u_avg*H/mu
    Re = (rho_f * u_avg * H) / max(1.0e-16, mu_f)
    print(f"[setup] H={H:.3e} m, u_avg={u_avg:.3e} m/s -> Re={Re:.3g}", flush=True)

    dt_c = Constant(dt_val)

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
        theta=theta,
        rho_f=Constant(rho_f),
        mu_f=Constant(mu_f),
        kappa_inv=Constant(float(args.kappa_inv)),
        mu_b_model="mu",
        solid_model="linear",
        mu_s=Constant(mu_s),
        lambda_s=Constant(lam_s),
        solid_visco_eta=float(args.mu_b),
        # Extension penalties so (u,vS) remain well-posed in the free fluid when alpha is small.
        gamma_u=float(args.gamma_u),
        u_extension_mode=str(args.u_extension),
        gamma_u_pin=float(args.gamma_u_pin),
        # Disable transport/kinetics (fields are frozen anyway).
        D_phi=0.0,
        gamma_phi=float(args.gamma_phi),
        D_alpha=0.0,
        D_S=0.0,
        mu_max=0.0,
        k_g=0.0,
        k_d=0.0,
        k_det=0.0,
        dim=2,
    )

    # ------------------------------------------------------------------
    # Boundary conditions (channel)
    # ------------------------------------------------------------------
    def ramp(t: float) -> float:
        t = float(t)
        if t <= 0.0:
            return 0.0
        if t >= t_ramp:
            return 1.0
        return t / t_ramp

    def inflow_vx(_x, y, t):
        yy = float(y) / H
        return float(Umax * ramp(t) * 4.0 * yy * (1.0 - yy))

    bcs = []
    bcs.append(BoundaryCondition("v_x", "dirichlet", "left", inflow_vx))
    bcs.append(BoundaryCondition("v_y", "dirichlet", "left", lambda x, y, t: 0.0))
    for tag in ("bottom", "top"):
        bcs.append(BoundaryCondition("v_x", "dirichlet", tag, lambda x, y, t: 0.0))
        bcs.append(BoundaryCondition("v_y", "dirichlet", tag, lambda x, y, t: 0.0))

    # Clamp the base (biofilm attached to the bottom wall).
    bcs.append(BoundaryCondition("u_x", "dirichlet", "bottom", lambda x, y, t: 0.0))
    bcs.append(BoundaryCondition("u_y", "dirichlet", "bottom", lambda x, y, t: 0.0))
    bcs.append(BoundaryCondition("vS_x", "dirichlet", "bottom", lambda x, y, t: 0.0))
    bcs.append(BoundaryCondition("vS_y", "dirichlet", "bottom", lambda x, y, t: 0.0))

    # Outlet: pin pressure.
    bcs.append(BoundaryCondition("p", "dirichlet", "right", lambda x, y, t: 0.0))
    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, lambda x, y: 0.0) for b in bcs]

    # ------------------------------------------------------------------
    # Output setup
    # ------------------------------------------------------------------
    out_dir = Path(str(args.out_dir))
    vtk_dir = out_dir / "vtk"
    vtk_every = int(args.vtk_every)
    out_dir.mkdir(parents=True, exist_ok=True)
    if vtk_every > 0:
        vtk_dir.mkdir(parents=True, exist_ok=True)

    # Initial export + time series header
    if vtk_every > 0:
        export_vtk(
            str(vtk_dir / "step=0000.vtu"),
            mesh,
            dh,
            {"v": v_n, "p": p_n, "vS": vS_n, "u": u_n, "phi": phi_n, "alpha": alpha_n, "S": S_n},
        )

    ts_path = out_dir / "timeseries.csv"
    with ts_path.open("w", encoding="utf-8") as f_ts:
        f_ts.write("t_s,dx_line1_m,dx_line2_m,dx_line3_m\n")
        f_ts.write("0.0,0.0,0.0,0.0\n")

    def _append_timeseries(t_s: float) -> None:
        a = np.asarray(alpha_k.nodal_values, dtype=float)
        x_now = [_x_alpha_half_on_y_line(alpha_xy, a, y_line=y0) for y0 in y_lines]
        dxs = [float(xn - xr) if (math.isfinite(xn) and math.isfinite(xr)) else float("nan") for xn, xr in zip(x_now, x_ref)]
        with ts_path.open("a", encoding="utf-8") as f_ts:
            f_ts.write(f"{float(t_s):.12e},{dxs[0]:.12e},{dxs[1]:.12e},{dxs[2]:.12e}\n")

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
    petsc_opts = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_atol": float(args.newton_tol),
        "snes_rtol": 0.0,
        "snes_max_it": int(args.max_it),
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    if os.getenv("PYCUTFEM_PETSC_MONITOR", "").strip().lower() in {"1", "true", "yes"}:
        petsc_opts.setdefault("snes_monitor", None)
        petsc_opts.setdefault("snes_converged_reason", None)
        petsc_opts.setdefault("ksp_converged_reason", None)
        petsc_opts.setdefault("ksp_monitor_short", None)

    solver = PetscSnesNewtonSolver(
        forms.residual_form,
        forms.jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=float(args.newton_tol), max_newton_iter=int(args.max_it)),
        quad_order=int(qdeg),
        backend=backend,
        petsc_options=petsc_opts,
    )

    def _on_dt_change(new_dt: float) -> None:
        dt_c.value = float(new_dt)

    def post_step_refiner(step, bcs_now, functions, prev_functions):
        # Called after an accepted Newton step and **before** promotion.
        _update_alpha_phi_from_refmap()
        step_no = int(getattr(solver, "_current_step_no", int(step)))
        t_now = float(getattr(solver, "_current_t", 0.0) + getattr(solver, "_current_dt", dt_val))
        _append_timeseries(t_now)
        if vtk_every > 0 and (step_no % vtk_every == 0):
            export_vtk(
                str(vtk_dir / f"step={step_no:04d}.vtu"),
                mesh,
                dh,
                {"v": v_k, "p": p_k, "vS": vS_k, "u": u_k, "phi": phi_k, "alpha": alpha_k, "S": S_k},
            )

    solver.solve_time_interval(
        functions=[v_k, p_k, vS_k, u_k, phi_k, alpha_k, S_k],
        prev_functions=[v_n, p_n, vS_n, u_n, phi_n, alpha_n, S_n],
        aux_functions={"dt": dt_c},
        time_params=TimeStepperParameters(dt=dt_val, final_time=float(args.t_final), max_steps=10_000, theta=theta, on_dt_change=_on_dt_change),
        post_step_refiner=post_step_refiner,
    )


if __name__ == "__main__":
    main()
