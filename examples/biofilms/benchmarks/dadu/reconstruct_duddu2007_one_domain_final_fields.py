"""
Reconstruct final (S, p) fields for a one-domain Duddu(2007) Fig.6 run from saved α snapshots.

Why
----
Many one-domain parameter sweeps are run with --skip-plots to save time. Those runs
still write `snaps_alpha.npz`, but do not store the final substrate/potential fields.
Since Duddu(2007) uses quasi-steady substrate and an elliptic potential solve at each
growth step, we can reconstruct the *final* (S, p) by solving:
  (1) substrate (quasi-steady) using the saved final α,
  (2) (p, vS) using the resulting S and the same α.

Inputs (in --results-dir)
-------------------------
- summary.json
- snaps_alpha.npz

Outputs (in --results-dir)
--------------------------
- final_fields.npz     (t_days, alpha, S, p)

Run (recommended)
-----------------
conda run --no-capture-output -n fenicsx python -u \
  examples/biofilms/benchmarks/dadu/reconstruct_duddu2007_one_domain_final_fields.py \
  --results-dir <OUTDIR> --backend cpp --linear-solver petsc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, NewtonSolver
from pycutfem.ufl.expressions import CellDiameter, Constant, Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import dInterface, dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad

from examples.biofilms.benchmarks.dadu.duddu2007_one_domain_growth_2d_fig6_example2 import (
    Duddu2007Params,
    _as_float_time,
    _biofilm_top_y,
    _set_inactive_fields,
    _tag_rectangle_boundaries,
    _update_phi_from_alpha,
)
from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms


def _load_alpha_snapshot(snaps_npz: Path, *, t_days: float | None) -> tuple[float, np.ndarray]:
    d = np.load(snaps_npz)
    t = np.asarray(d["t_days"], dtype=float).ravel()
    a = np.asarray(d["alpha"], dtype=float)
    if a.ndim != 2 or a.shape[0] != t.size:
        raise ValueError("Unexpected snaps_alpha.npz shapes.")
    if t_days is None:
        j = int(t.size - 1)
    else:
        j = int(np.argmin(np.abs(t - float(t_days))))
    return float(t[j]), np.asarray(a[j, :], dtype=float).copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, required=True)
    ap.add_argument("--t-days", type=float, default=float("nan"), help="Snapshot time to reconstruct (default: final).")
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--linear-solver", type=str, default="petsc", choices=("petsc", "scipy"))
    ap.add_argument("--newton-tol", type=float, default=1.0e-8)
    ap.add_argument("--max-it", type=int, default=30)
    ap.add_argument("--q", type=int, default=-1, help="Override quadrature order (default: use summary.json).")
    ap.add_argument("--dt-steady", type=float, default=1.0e6, help="Large dt used to approximate quasi-steady S (days).")
    args = ap.parse_args()

    outdir = Path(str(args.results_dir)).expanduser()
    summary_path = outdir / "summary.json"
    snaps_path = outdir / "snaps_alpha.npz"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not snaps_path.exists():
        raise FileNotFoundError(snaps_path)

    summary = json.loads(summary_path.read_text())
    L = float(summary["L_mm"])
    H = float(summary["H_mm"])
    nx = int(summary["nx"])
    ny = int(summary["ny"])
    qdeg = int(summary["q"]) if int(args.q) < 0 else int(args.q)

    t_req = None if not np.isfinite(float(args.t_days)) else float(args.t_days)
    t_snap, alpha_vals = _load_alpha_snapshot(snaps_path, t_days=t_req)
    print(f"[reconstruct] Using α snapshot at t={t_snap:.3f} d (requested={t_req}).")

    # --- mesh ---------------------------------------------------------------
    nodes, elems, _edges, corners = structured_quad(L, H, nx=nx, ny=ny, poly_order=2, offset=(0.0, 0.0))
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=2)
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

    # Spaces / trial/test functions
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

    # Functions at k and n
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

    functions = [v_k, p_k, vS_k, u_k, phi_k, alpha_k, S_k]
    prev_functions = [v_n, p_n, vS_n, u_n, phi_n, alpha_n, S_n]

    for vf in (v_k, vS_k, u_k, v_n, vS_n, u_n):
        vf.nodal_values[:] = 0.0
    for sf in (p_k, p_n):
        sf.nodal_values[:] = 0.0

    # Load alpha
    alpha_vals = np.clip(np.asarray(alpha_vals, dtype=float).ravel(), 0.0, 1.0)
    if alpha_vals.size != np.asarray(alpha_n.nodal_values).size:
        raise ValueError(f"alpha snapshot size mismatch: {alpha_vals.size} vs {np.asarray(alpha_n.nodal_values).size}")
    alpha_n.nodal_values[:] = alpha_vals
    alpha_k.nodal_values[:] = alpha_vals

    # Frozen porosity proxy from alpha
    phi_b = float(summary["phi_b"])
    _update_phi_from_alpha(
        phi=phi_n,
        alpha=alpha_n,
        phi_b=phi_b,
        mode=str(summary.get("phi_update", "mix")),
        alpha0=float(summary.get("phi_alpha0", 0.1)),
        alpha_width=float(summary.get("phi_alpha_width", 0.05)),
    )
    phi_k.nodal_values[:] = phi_n.nodal_values[:]

    # Substrate init
    Sbar = float(summary["Sbar"])
    S_n.set_values_from_function(lambda x, y: float(Sbar))
    S_k.nodal_values[:] = S_n.nodal_values[:]

    # Time constants used by one_domain forms (updated per block).
    dt_c = Constant(float(args.dt_steady))
    dx_q = dx(metadata={"q": int(qdeg)})

    # Kinetics mapping (same as the main driver).
    kin = Duddu2007Params()
    one_m_phi_b = 1.0 - float(phi_b)
    desired_uptake = kin.f_active * (kin.qhat0 + kin.g * kin.f_D * kin.b)
    mu_max = float(desired_uptake) * float(kin.Y_xO) / float(one_m_phi_b)
    mu_max *= float(summary.get("mu_max_scale", 1.0))

    # Growth source for the constraint.
    divU_k = kin.divU(S_k)
    sv_mode = str(summary.get("s_v_mode", "auto")).strip().lower()
    if sv_mode == "auto":
        adv_key = str(summary.get("alpha_advect_with", "vS")).strip().lower()
        sv_mode = "divu" if adv_key.startswith("mix") else "bdivu"
    if sv_mode in {"divu", "u"}:
        s_v = divU_k
    elif sv_mode in {"bdivu", "b*divu", "b"}:
        s_v = ((-phi_k) + Constant(1.0)) * divU_k
    else:
        raise ValueError(f"Unknown s_v_mode in summary.json: {sv_mode!r}")
    ds_v = Constant(0.0)

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
        theta=1.0,
        rho_f=Constant(0.0),
        mu_f=Constant(float(summary.get("mu_f", 1.0))),
        kappa_inv=Constant(float(summary.get("kappa_inv", 8.0))),
        mu_b_model="phi_mu",
        solid_model="linear",
        mu_s=Constant(1.0e-8),
        lambda_s=Constant(1.0e-8),
        solid_visco_eta=0.0,
        include_skeleton_acceleration=False,
        # freeze phi equation (phi is inactive)
        D_phi=0.0,
        gamma_phi=0.0,
        # alpha (kept only as coefficient here)
        D_alpha=float(summary.get("D_alpha", 0.0)),
        alpha_advect_with=str(summary.get("alpha_advect_with", "vS")),
        alpha_advection_form="advective",
        alpha_cahn_M=float(summary.get("ac_M", 0.0)),
        alpha_cahn_gamma=float(summary.get("ac_gamma", 0.0)),
        alpha_cahn_eps=float(summary.get("eps_alpha_mm", 0.01)),
        alpha_cahn_mobility=str(summary.get("ac_mobility", "constant")),
        alpha_cahn_mobility_floor=float(summary.get("ac_mobility_floor", 0.0)),
        alpha_cahn_conservative_mode="eliminate",
        alpha_supg=float(summary.get("alpha_supg", 0.0)),
        # substrate
        D_S=float(summary.get("D_S_mm2_per_day", 120.0)),
        substrate_reaction_scheme="implicit",
        substrate_diffusion_scheme="implicit",
        mu_max=float(mu_max),
        K_S=float(kin.K0),
        k_g=0.0,
        k_d=0.0,
        Y=float(kin.Y_xO),
        rho_s_star=float(kin.rho_x),
        k_det=0.0,
        s_v=s_v,
        ds_v=ds_v,
        D_det_prev=Constant(0.0),
        # vS extension
        gamma_vS=float(summary.get("gamma_vS", 0.0)),
        vS_extension_mode=str(summary.get("vS_ext_mode", "l2")),
        gamma_vS_pin=float(summary.get("gamma_vS_pin", 0.0)),
        gamma_vS_pin_power=int(summary.get("gamma_vS_pin_power", 2)),
    )

    # Moving substrate Dirichlet line via penalty on dInterface (CutFEM).
    Ls = float(summary.get("Ls_mm", 0.1))
    y_top = _biofilm_top_y(dh=dh, alpha=alpha_n, alpha_half=0.5)
    y_D = float(min(float(H) - 1.0e-12, y_top + float(Ls) + 1.0e-10))
    ls_Sd = AffineLevelSet(0.0, 1.0, -float(y_D))
    dGammaS = dInterface(level_set=ls_Sd, metadata={"q": int(qdeg), "linear_interface": True})
    penS = Constant(float(summary.get("S_penalty", 1.0e6)))
    Sbar_c = Constant(float(Sbar))
    h = CellDiameter()
    r_Spen = (penS / h) * (S_k - Sbar_c) * S_test * dGammaS
    a_Spen = (penS / h) * dS * S_test * dGammaS

    # (p,vS) p-out penalty (keeps p DOFs in the pure fluid well-posed).
    m_pow = int(max(1, int(summary.get("gamma_p_out_power", 2))))
    one_m_alpha = (-alpha_k) + Constant(1.0)
    w_p_out = one_m_alpha
    for _ in range(m_pow - 1):
        w_p_out = w_p_out * one_m_alpha
    gamma_p_out = Constant(float(summary.get("gamma_p_out", 1.0e6)))
    r_p_out = gamma_p_out * w_p_out * p_k * q_test * dx_q
    a_p_out = gamma_p_out * w_p_out * dp * q_test * dx_q

    # Boundary conditions for (p, vS) (no fluid in this reconstruction).
    bc_vSx_left = BoundaryCondition("vS_x", "dirichlet", "left", _as_float_time(lambda x, y, t: 0.0))
    bc_vSx_left_h = BoundaryCondition("vS_x", "dirichlet", "left", (lambda x, y: 0.0))
    bc_vSx_right = BoundaryCondition("vS_x", "dirichlet", "right", _as_float_time(lambda x, y, t: 0.0))
    bc_vSx_right_h = BoundaryCondition("vS_x", "dirichlet", "right", (lambda x, y: 0.0))
    bc_vSy_bottom = BoundaryCondition("vS_y", "dirichlet", "bottom", _as_float_time(lambda x, y, t: 0.0))
    bc_vSy_bottom_h = BoundaryCondition("vS_y", "dirichlet", "bottom", (lambda x, y: 0.0))
    bc_p_top = BoundaryCondition("p", "dirichlet", "top", _as_float_time(lambda x, y, t: 0.0))
    bc_p_top_h = BoundaryCondition("p", "dirichlet", "top", (lambda x, y: 0.0))

    bcs_pvS = [bc_vSx_left, bc_vSx_right, bc_vSy_bottom, bc_p_top]
    bcs_pvS_homog = [bc_vSx_left_h, bc_vSx_right_h, bc_vSy_bottom_h, bc_p_top_h]

    # --- solvers -----------------------------------------------------------
    newton_tol = float(args.newton_tol)
    max_it = int(args.max_it)
    lin_backend = str(args.linear_solver)
    backend = str(args.backend)

    # Substrate: only S active
    inactive_S = ["v_x", "v_y", "p", "vS_x", "vS_y", "u_x", "u_y", "phi", "alpha"]
    _set_inactive_fields(dh, inactive_S)
    solver_S = NewtonSolver(
        forms.r_substrate + r_Spen,
        forms.a_substrate + a_Spen,
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(
            newton_tol=newton_tol,
            newton_rtol=0.0,
            max_newton_iter=max_it,
            print_level=0,
            line_search=True,
            ls_mode="dealii",
        ),
        lin_params=LinearSolverParameters(backend=lin_backend),
        quad_order=qdeg,
        backend=backend,
    )

    # (p,vS): only p and vS active
    inactive_pvS = ["u_x", "u_y", "phi", "alpha", "S", "v_x", "v_y"]
    _set_inactive_fields(dh, inactive_pvS)
    r_pvS = forms.r_mass + forms.r_skeleton + r_p_out
    a_pvS = forms.a_mass + forms.a_skeleton + a_p_out
    solver_pvS = NewtonSolver(
        r_pvS,
        a_pvS,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs_pvS,
        bcs_homog=bcs_pvS_homog,
        newton_params=NewtonParameters(
            newton_tol=newton_tol,
            newton_rtol=0.0,
            max_newton_iter=max_it,
            print_level=0,
            line_search=False,
            ls_mode="dealii",
        ),
        lin_params=LinearSolverParameters(backend=lin_backend),
        quad_order=qdeg,
        backend=backend,
    )

    # --- solve substrate ---------------------------------------------------
    dt_c.value = float(args.dt_steady)
    solver_S._current_step_no = 0
    solver_S._current_t = float(t_snap)
    solver_S._current_dt = float(args.dt_steady)
    solver_S._newton_loop(functions, prev_functions, {"dt": dt_c}, bcs_now=[])
    S_k.nodal_values[:] = np.maximum(np.asarray(S_k.nodal_values, dtype=float), 0.0)

    # --- solve pvS ---------------------------------------------------------
    dt_run = float(summary.get("dt_days", 0.2))
    dt_c.value = float(dt_run)
    solver_pvS._current_step_no = 0
    solver_pvS._current_t = float(t_snap)
    solver_pvS._current_dt = float(dt_run)
    bcs_now = solver_pvS._freeze_bcs(bcs_pvS, float(t_snap))
    dh.apply_bcs(bcs_now, *functions)
    solver_pvS._newton_loop(functions, prev_functions, {"dt": dt_c}, bcs_now)

    # Promote to *_n for output consistency
    S_n.nodal_values[:] = S_k.nodal_values[:]
    p_n.nodal_values[:] = p_k.nodal_values[:]

    np.savez_compressed(
        outdir / "final_fields.npz",
        t_days=float(t_snap),
        alpha=np.asarray(alpha_n.nodal_values, dtype=np.float32),
        S=np.asarray(S_n.nodal_values, dtype=np.float32),
        p=np.asarray(p_n.nodal_values, dtype=np.float32),
    )
    print(f"- Wrote {outdir/'final_fields.npz'}")


if __name__ == "__main__":
    main()

