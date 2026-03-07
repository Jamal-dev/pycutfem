import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Avoid extremely verbose Numba debug dumps if the environment enables them.
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
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
from pycutfem.ufl.analytic import Analytic
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
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.measures import dx
from examples.utils.biofilm.mms_moving_interface import BiofilmMovingInterfaceMMS
from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms
from pycutfem.utils.meshgen import structured_quad


def _tag_unit_square_boundaries(mesh: Mesh, *, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - 1.0) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - 1.0) <= tol,
        }
    )


def _as_float_time(fn):
    return lambda x, y, t: float(np.asarray(fn(np.asarray(x), np.asarray(y), float(t))).reshape(()))


def _eoc(prev_h: float, h: float, prev_err: float, err: float) -> float:
    if not (prev_h > 0.0 and h > 0.0 and prev_err > 0.0 and err > 0.0):
        return float("nan")
    return float(math.log(prev_err / err) / math.log(prev_h / h))


def _run_one(
    *,
    nx: int,
    ny: int,
    qdeg: int,
    qerr: int,
    dt_val: float,
    nsteps: int,
    backend: str,
    newton_tol: float,
    max_it: int,
    # MMS parameters
    h0: float,
    V_det: float,
    phi_b: float,
    S0: float,
    a: float,
    omega: float,
    eps: float,
    # model/forcing parameters
    rho_f: float,
    mu_f: float,
    kappa_inv: float,
    D_phi: float,
    gamma_phi: float,
    gamma_u: float,
    D_alpha: float,
    k_det: float,
    eta_n: float,
    vtk_every: int = 0,
    vtk_dir=None,
):
    theta = 1.0
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=int(nx), ny=int(ny), poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    _tag_unit_square_boundaries(mesh)

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
            "X": 1,
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
    dX = TrialFunction("X", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    vS_test = VectorTestFunction(space=VS, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    S_test = TestFunction("S", dof_handler=dh)
    X_test = TestFunction("X", dof_handler=dh)

    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    vS_k = VectorFunction("vS_k", ["vS_x", "vS_y"], dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    S_k = Function("S_k", "S", dof_handler=dh)
    X_k = Function("X_k", "X", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    vS_n = VectorFunction("vS_n", ["vS_x", "vS_y"], dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    S_n = Function("S_n", "S", dof_handler=dh)
    X_n = Function("X_n", "X", dof_handler=dh)

    mms = BiofilmMovingInterfaceMMS(
        h0=float(h0),
        V_det=float(V_det),
        eps=float(eps),
        phi_b=float(phi_b),
        a=float(a),
        omega=float(omega),
        S0=float(S0),
        rho_f=float(rho_f),
        mu_f=float(mu_f),
        kappa_inv=float(kappa_inv),
        D_phi=float(D_phi),
        gamma_phi=float(gamma_phi),
        D_alpha=float(D_alpha),
        k_det=float(k_det),
        eta_n=float(eta_n),
        t_n=0.0,
        dt=float(dt_val),
    )

    t0 = 0.0
    v_n.set_values_from_function(lambda x, y: mms.v(x, y, t0))
    vS_n.nodal_values.fill(0.0)
    u_n.set_values_from_function(lambda x, y: mms.u(x, y, t0))
    p_n.set_values_from_function(lambda x, y: float(mms.p(x, y, t0)))
    phi_n.set_values_from_function(lambda x, y: float(mms.phi(x, y, t0)))
    alpha_n.set_values_from_function(lambda x, y: float(mms.alpha(x, y, t0)))
    S_n.set_values_from_function(lambda x, y: float(mms.S(x, y, t0)))
    X_n.set_values_from_function(lambda x, y: float(mms.X(x, y, t0)))

    dt_c = Constant(float(dt_val))

    f_v = Analytic(lambda x, y: mms.f_v(x, y), degree=8)
    f_u = Analytic(lambda x, y: mms.f_u(x, y), degree=8)
    s_v = Analytic(lambda x, y: mms.s_v(x, y), degree=4)
    f_phi = Analytic(lambda x, y: mms.f_phi(x, y), degree=8)
    f_alpha = Analytic(lambda x, y: mms.f_alpha(x, y), degree=8)
    f_S = Analytic(lambda x, y: mms.f_S(x, y), degree=6)
    D_det_prev = Analytic(lambda x, y: mms.D_det_prev(x, y), degree=8)
    f_X = Analytic(lambda x, y: mms.f_X(x, y), degree=8)

    forms = build_biofilm_one_domain_forms(
        v_k=v_k,
        p_k=p_k,
        vS_k=vS_k,
        u_k=u_k,
        phi_k=phi_k,
        alpha_k=alpha_k,
        S_k=S_k,
        X_k=X_k,
        v_n=v_n,
        p_n=p_n,
        vS_n=vS_n,
        u_n=u_n,
        phi_n=phi_n,
        alpha_n=alpha_n,
        S_n=S_n,
        X_n=X_n,
        dv=dv,
        dp=dp,
        dvS=dvS,
        du=du,
        dphi=dphi,
        dalpha=dalpha,
        dS=dS,
        dX=dX,
        v_test=v_test,
        q_test=q_test,
        vS_test=vS_test,
        u_test=u_test,
        phi_test=phi_test,
        alpha_test=alpha_test,
        S_test=S_test,
        X_test=X_test,
        dx=dx(metadata={"q": int(qdeg)}),
        dt=dt_c,
        theta=theta,
        rho_f=Constant(float(rho_f)),
        mu_f=Constant(float(mu_f)),
        kappa_inv=Constant(float(kappa_inv)),
        mu_s=Constant(1.0),
        lambda_s=Constant(1.0),
        D_phi=float(D_phi),
        gamma_phi=float(gamma_phi),
        gamma_u=float(gamma_u),
        D_alpha=float(D_alpha),
        D_S=0.0,
        D_X=0.1,
        mu_max=0.0,
        K_S=1.0,
        k_g=0.0,
        k_d=0.0,
        Y=1.0,
        k_det=float(k_det),
        f_v=f_v,
        f_u=f_u,
        s_v=s_v,
        f_phi=f_phi,
        f_alpha=f_alpha,
        f_S=f_S,
        f_X=f_X,
        D_det_prev=D_det_prev,
    )

    bcs = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                BoundaryCondition("v_x", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.v(x, y, t)[..., 0])),
                BoundaryCondition("v_y", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.v(x, y, t)[..., 1])),
                BoundaryCondition("p", "dirichlet", tag, _as_float_time(mms.p)),
                BoundaryCondition("vS_x", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)),
                BoundaryCondition("vS_y", "dirichlet", tag, _as_float_time(lambda x, y, t: 0.0)),
                BoundaryCondition("u_x", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.u(x, y, t)[..., 0])),
                BoundaryCondition("u_y", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.u(x, y, t)[..., 1])),
                BoundaryCondition("phi", "dirichlet", tag, _as_float_time(mms.phi)),
                BoundaryCondition("alpha", "dirichlet", tag, _as_float_time(mms.alpha)),
                BoundaryCondition("S", "dirichlet", tag, _as_float_time(mms.S)),
                BoundaryCondition("X", "dirichlet", tag, _as_float_time(mms.X)),
            ]
        )
    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, (lambda x, y: 0.0)) for b in bcs]

    step_rows = []
    solver_ref = {}
    vtk_every = int(vtk_every)
    vtk_dir_run = None
    if vtk_every > 0:
        if vtk_dir is None:
            raise ValueError("vtk_dir is required when vtk_every > 0.")
        vtk_dir_run = Path(vtk_dir) / f"nx={int(nx)}"
        vtk_dir_run.mkdir(parents=True, exist_ok=True)

    def _preproc_cb(_funcs):
        solver = solver_ref.get("solver")
        if solver is None:
            return
        mms.set_step_time(t_n=float(solver._current_t), dt=float(solver._current_dt))

    def _post_timeloop_cb(_funcs):
        solver = solver_ref.get("solver")
        if solver is None:
            return
        step_no = len(step_rows) + 1
        t_err = float(solver._current_t + solver._current_dt)
        err_v = dh.l2_error(
            v_k,
            exact={"v_x": lambda x, y: mms.v(x, y, t_err)[..., 0], "v_y": lambda x, y: mms.v(x, y, t_err)[..., 1]},
            fields=["v_x", "v_y"],
            quad_order=int(qerr),
            relative=False,
        )
        err_phi = dh.l2_error(phi_k, exact={"phi": lambda x, y: mms.phi(x, y, t_err)}, fields=["phi"], quad_order=int(qerr), relative=False)
        err_alpha = dh.l2_error(
            alpha_k, exact={"alpha": lambda x, y: mms.alpha(x, y, t_err)}, fields=["alpha"], quad_order=int(qerr), relative=False
        )
        err_X = dh.l2_error(X_k, exact={"X": lambda x, y: mms.X(x, y, t_err)}, fields=["X"], quad_order=int(qerr), relative=False)
        step_rows.append({"dt": float(solver._current_dt), "err_v": float(err_v), "err_phi": float(err_phi), "err_alpha": float(err_alpha)})
        step_rows[-1]["err_X"] = float(err_X)
        if vtk_dir_run is not None and (step_no % vtk_every == 0):
            export_vtk(
                str(vtk_dir_run / f"step={step_no:04d}.vtu"),
                mesh,
                dh,
                {"v": v_k, "p": p_k, "vS": vS_k, "u": u_k, "phi": phi_k, "alpha": alpha_k, "S": S_k, "X": X_k},
            )

    solver = NewtonSolver(
        forms.residual_form,
        forms.jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=float(newton_tol), max_newton_iter=int(max_it)),
        quad_order=int(qdeg),
        backend=str(backend),
        preproc_cb=_preproc_cb,
        postproc_timeloop_cb=_post_timeloop_cb,
    )
    solver_ref["solver"] = solver

    solver.solve_time_interval(
        functions=[v_k, p_k, vS_k, u_k, phi_k, alpha_k, S_k, X_k],
        prev_functions=[v_n, p_n, vS_n, u_n, phi_n, alpha_n, S_n, X_n],
        aux_functions={"dt": dt_c},
        time_params=TimeStepperParameters(dt=float(dt_val), final_time=float(nsteps) * float(dt_val), max_steps=int(nsteps), theta=theta, t0=t0),
    )

    if not step_rows:
        raise RuntimeError("No time steps executed.")

    def _agg_max(key: str) -> float:
        return float(max(r[key] for r in step_rows))

    def _agg_l2(key: str) -> float:
        return float(math.sqrt(sum(float(r["dt"]) * float(r[key]) * float(r[key]) for r in step_rows)))

    h = 1.0 / float(nx)
    return {
        "nx": int(nx),
        "h": float(h),
        "eps": float(eps),
        "ev_max": _agg_max("err_v"),
        "ev_l2t": _agg_l2("err_v"),
        "ephi_max": _agg_max("err_phi"),
        "ephi_l2t": _agg_l2("err_phi"),
        "ealpha_max": _agg_max("err_alpha"),
        "ealpha_l2t": _agg_l2("err_alpha"),
        "eX_max": _agg_max("err_X"),
        "eX_l2t": _agg_l2("err_X"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="h-convergence for the moving-interface (detachment-like) MMS time test.")
    ap.add_argument("--nx-list", type=str, default="8,16,32", help="Comma-separated nx values (ny=nx).")
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--q-error", type=int, default=12)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--nsteps", type=int, default=5)
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--newton-tol", type=float, default=1.0e-10)
    ap.add_argument("--max-it", type=int, default=30)
    ap.add_argument("--convergence", action="store_true", help="Save log-log convergence plot as a PNG.")
    ap.add_argument("--outdir", type=str, default="examples/biofilms/results/moving_interface", help="Directory for saving CSV/LaTeX tables (and plots).")
    ap.add_argument("--vtk-every", type=int, default=0, help="Write VTK every N accepted steps (0 disables).")

    # MMS parameters
    ap.add_argument("--h0", type=float, default=0.4)
    ap.add_argument("--V-det", type=float, default=0.2)
    ap.add_argument("--phi-b", type=float, default=0.6)
    ap.add_argument("--S0", type=float, default=0.5)
    ap.add_argument("--a", type=float, default=0.2)
    ap.add_argument("--omega", type=float, default=2.0 * math.pi)
    ap.add_argument(
        "--eps",
        type=float,
        default=None,
        help=(
            "Diffuse-interface thickness epsilon. If omitted, eps is selected by --eps-mode "
            "('fixed' uses eps-factor*h_coarsest; 'scaled' uses eps-factor*h on each mesh)."
        ),
    )
    ap.add_argument("--eps-factor", type=float, default=2.0)
    ap.add_argument(
        "--eps-mode",
        type=str,
        default="fixed",
        choices=("fixed", "scaled"),
        help=(
            "How to choose epsilon when --eps is omitted. "
            "'fixed': eps = eps-factor*h_coarsest and held constant across meshes (standard h-convergence). "
            "'scaled': eps = eps-factor*h on each mesh (interface sharpens with refinement; expect reduced EOC for alpha/phi)."
        ),
    )

    # model/forcing parameters
    ap.add_argument("--rho-f", type=float, default=1.0)
    ap.add_argument("--mu-f", type=float, default=1.0e-2)
    ap.add_argument("--kappa-inv", type=float, default=10.0)
    ap.add_argument("--D-phi", type=float, default=0.1)
    ap.add_argument("--gamma-phi", type=float, default=1.0)
    ap.add_argument(
        "--gamma-u",
        type=float,
        default=None,
        help=(
            "Extension penalty factor for skeleton displacement in the free-fluid region (stabilizes u as alpha->0). "
            "In the core model this is used as gamma_u/h^2 (MeshSize scaling). "
            "If omitted, uses gamma_u = gamma_u_factor."
        ),
    )
    ap.add_argument(
        "--gamma-u-factor",
        type=float,
        default=1.0,
        help="Default gamma_u factor used when --gamma-u is omitted (core model uses gamma_u/h^2).",
    )
    ap.add_argument("--D-alpha", type=float, default=0.1)
    ap.add_argument("--k-det", type=float, default=0.2)
    ap.add_argument("--eta-n", type=float, default=1.0e-12)
    args = ap.parse_args()

    nx_list = [int(s.strip()) for s in str(args.nx_list).split(",") if s.strip()]
    if not nx_list:
        raise ValueError("Empty --nx-list.")

    eps_mode = str(getattr(args, "eps_mode", "fixed")).strip().lower()
    if eps_mode not in {"fixed", "scaled"}:
        raise ValueError(f"Unknown --eps-mode {args.eps_mode!r}. Use 'fixed' or 'scaled'.")
    nx_coarse = int(min(nx_list))
    h_coarse = 1.0 / float(nx_coarse)
    if args.eps is not None:
        eps_mode_eff = "fixed"
        eps_fixed = float(args.eps)
        print(f"[info] Using fixed eps={eps_fixed:g} across meshes (from --eps).")
    elif eps_mode == "fixed":
        eps_mode_eff = "fixed"
        eps_fixed = float(args.eps_factor) * float(h_coarse)
        print(
            f"[info] --eps omitted; using fixed eps=eps-factor*h_coarsest = {float(args.eps_factor):g}*{h_coarse:g} = {eps_fixed:g}."
        )
    else:
        eps_mode_eff = "scaled"
        eps_fixed = None
        print(
            "[info] --eps omitted; using eps = eps-factor*h on each mesh (scaled). "
            "This is not a fixed-PDE h-convergence study; expect reduced EOC for alpha/phi and coupled fields as eps shrinks."
        )

    outdir = Path(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"biofilm_mms_moving_interface_backend={args.backend}_dt={float(args.dt):g}_nsteps={int(args.nsteps)}_epsmode={eps_mode_eff}"
    if eps_mode_eff == "fixed":
        stem += f"_eps={float(eps_fixed):g}"
    else:
        stem += f"_epsfac={float(args.eps_factor):g}"
    vtk_dir = outdir / "vtk" / stem if int(args.vtk_every) > 0 else None

    rows = []
    for nx in nx_list:
        h = 1.0 / float(nx)
        eps = float(eps_fixed) if eps_mode_eff == "fixed" else float(args.eps_factor) * h
        gamma_u = float(args.gamma_u) if args.gamma_u is not None else float(args.gamma_u_factor)
        rows.append(
            _run_one(
                nx=nx,
                ny=nx,
                qdeg=int(args.q),
                qerr=int(args.q_error),
                dt_val=float(args.dt),
                nsteps=int(args.nsteps),
                backend=str(args.backend),
                newton_tol=float(args.newton_tol),
                max_it=int(args.max_it),
                h0=float(args.h0),
                V_det=float(args.V_det),
                phi_b=float(args.phi_b),
                S0=float(args.S0),
                a=float(args.a),
                omega=float(args.omega),
                eps=float(eps),
                rho_f=float(args.rho_f),
                mu_f=float(args.mu_f),
                kappa_inv=float(args.kappa_inv),
                D_phi=float(args.D_phi),
                gamma_phi=float(args.gamma_phi),
                gamma_u=float(gamma_u),
                D_alpha=float(args.D_alpha),
                k_det=float(args.k_det),
                eta_n=float(args.eta_n),
                vtk_every=int(args.vtk_every),
                vtk_dir=vtk_dir,
            )
        )

    print(
        f"\nMoving-interface MMS convergence | backend={args.backend} | dt={float(args.dt):g} | nsteps={int(args.nsteps)} | eps-mode={eps_mode_eff}"
    )

    recs: list[dict] = []
    for i, r in enumerate(rows):
        prev = rows[i - 1] if i > 0 else None
        recs.append(
            {
                "nx": int(r["nx"]),
                "h": float(r["h"]),
                "eps": float(r["eps"]),
                "ev_max": float(r["ev_max"]),
                "eoc_ev_max": float("nan") if prev is None else _eoc(prev["h"], r["h"], prev["ev_max"], r["ev_max"]),
                "ev_l2t": float(r["ev_l2t"]),
                "eoc_ev_l2t": float("nan") if prev is None else _eoc(prev["h"], r["h"], prev["ev_l2t"], r["ev_l2t"]),
                "ephi_max": float(r["ephi_max"]),
                "eoc_ephi_max": float("nan") if prev is None else _eoc(prev["h"], r["h"], prev["ephi_max"], r["ephi_max"]),
                "ephi_l2t": float(r["ephi_l2t"]),
                "eoc_ephi_l2t": float("nan") if prev is None else _eoc(prev["h"], r["h"], prev["ephi_l2t"], r["ephi_l2t"]),
                "ealpha_max": float(r["ealpha_max"]),
                "eoc_ealpha_max": float("nan") if prev is None else _eoc(prev["h"], r["h"], prev["ealpha_max"], r["ealpha_max"]),
                "ealpha_l2t": float(r["ealpha_l2t"]),
                "eoc_ealpha_l2t": float("nan") if prev is None else _eoc(prev["h"], r["h"], prev["ealpha_l2t"], r["ealpha_l2t"]),
                "eX_max": float(r["eX_max"]),
                "eoc_eX_max": float("nan") if prev is None else _eoc(prev["h"], r["h"], prev["eX_max"], r["eX_max"]),
                "eX_l2t": float(r["eX_l2t"]),
                "eoc_eX_l2t": float("nan") if prev is None else _eoc(prev["h"], r["h"], prev["eX_l2t"], r["eX_l2t"]),
            }
        )

    df = pd.DataFrame.from_records(recs)
    print("\nConvergence table (pandas):")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 160):
        print(df)

    latex = df.to_latex(index=False, float_format="%.3e", na_rep="-")
    print("\nLaTeX table (copy/paste):\n")
    print(latex)

    out_csv = outdir / f"{stem}.csv"
    out_tex = outdir / f"{stem}.tex"
    df.to_csv(out_csv, index=False)
    out_tex.write_text(latex, encoding="utf-8")
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_tex}")

    if bool(args.convergence):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] --convergence requested but matplotlib is unavailable: {exc}")
            return

        hs = np.asarray([r["h"] for r in rows], dtype=float)
        y_vmax = np.asarray([r["ev_max"] for r in rows], dtype=float)
        y_vl2t = np.asarray([r["ev_l2t"] for r in rows], dtype=float)
        y_al2t = np.asarray([r["ealpha_l2t"] for r in rows], dtype=float)
        y_Xl2t = np.asarray([r["eX_l2t"] for r in rows], dtype=float)

        fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)

        def _label(name: str, vals: np.ndarray) -> str:
            if hs.size >= 2 and np.all(vals > 0.0):
                slope = float(np.polyfit(np.log(hs), np.log(vals), 1)[0])
                return f"{name} (p≈{slope:.2f})"
            return name

        ax.loglog(hs, y_vmax, marker="o", linewidth=1.5, label=_label("e_v(max)", y_vmax))
        ax.loglog(hs, y_vl2t, marker="o", linewidth=1.5, label=_label("e_v(L2t)", y_vl2t))
        ax.loglog(hs, y_al2t, marker="o", linewidth=1.5, label=_label("e_alpha(L2t)", y_al2t))
        ax.loglog(hs, y_Xl2t, marker="o", linewidth=1.5, label=_label("e_X(L2t)", y_Xl2t))

        ax.set_xlabel("h")
        ax.set_ylabel("Error")
        ax.grid(True, which="both", linestyle=":", linewidth=0.7)
        ax.legend(loc="best", fontsize=9)
        ax.set_title(f"Moving-interface MMS convergence (backend={args.backend}, dt={float(args.dt):g}, nsteps={int(args.nsteps)})")

        out = outdir / f"{stem}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f"\nSaved convergence plot to {out}")


if __name__ == "__main__":
    main()
