import math
import os

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction as UFLTestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
)
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad

from examples.utils.biofilm.mms_one_domain_convergence import build_biofilm_one_domain_mms_trig_step
from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms


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


def _solve_one(*, nx: int, qdeg: int, qerr: int, dt_val: float, theta: float, newton_tol: float, max_it: int, k_det: float, mms):
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=int(nx), ny=int(nx), poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    _tag_unit_square_boundaries(mesh)

    alpha_cahn_conservative = bool(getattr(mms, "alpha_cahn_conservative", False))
    alpha_ch_enabled = bool(getattr(mms, "alpha_ch_M", 0.0) or 0.0) and bool(getattr(mms, "alpha_ch_gamma", 0.0) or 0.0)
    if alpha_ch_enabled and alpha_cahn_conservative:
        raise ValueError("MMS setup does not support alpha_cahn_conservative together with Cahn–Hilliard (alpha_ch_*).")

    field_specs = {
        "v_x": 2,
        "v_y": 2,
        "p": 1,
        "u_x": 2,
        "u_y": 2,
        "phi": 1,
        "alpha": 1,
        **({"mu_alpha": 1} if alpha_ch_enabled else {}),
        "S": 1,
        "X": 1,
    }
    if alpha_cahn_conservative:
        field_specs["lambda_alpha"] = ":number:"

    me = MixedElement(mesh, field_specs=field_specs)
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    dv = VectorTrialFunction(space=V, dof_handler=dh)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    dp = TrialFunction("p", dof_handler=dh)
    dphi = TrialFunction("phi", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    dmu_alpha = TrialFunction("mu_alpha", dof_handler=dh) if alpha_ch_enabled else None
    dlambda_alpha = TrialFunction("lambda_alpha", dof_handler=dh) if alpha_cahn_conservative else None
    dS = TrialFunction("S", dof_handler=dh)
    dX = TrialFunction("X", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = UFLTestFunction("p", dof_handler=dh)
    phi_test = UFLTestFunction("phi", dof_handler=dh)
    alpha_test = UFLTestFunction("alpha", dof_handler=dh)
    mu_alpha_test = UFLTestFunction("mu_alpha", dof_handler=dh) if alpha_ch_enabled else None
    lambda_alpha_test = UFLTestFunction("lambda_alpha", dof_handler=dh) if alpha_cahn_conservative else None
    S_test = UFLTestFunction("S", dof_handler=dh)
    X_test = UFLTestFunction("X", dof_handler=dh)

    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    mu_alpha_k = Function("mu_alpha_k", "mu_alpha", dof_handler=dh) if alpha_ch_enabled else None
    lambda_alpha_k = Function("lambda_alpha_k", "lambda_alpha", dof_handler=dh) if alpha_cahn_conservative else None
    S_k = Function("S_k", "S", dof_handler=dh)
    X_k = Function("X_k", "X", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    mu_alpha_n = Function("mu_alpha_n", "mu_alpha", dof_handler=dh) if alpha_ch_enabled else None
    lambda_alpha_n = Function("lambda_alpha_n", "lambda_alpha", dof_handler=dh) if alpha_cahn_conservative else None
    S_n = Function("S_n", "S", dof_handler=dh)
    X_n = Function("X_n", "X", dof_handler=dh)

    v_n.set_values_from_function(lambda x, y: mms.v(x, y, mms.t_n))
    u_n.set_values_from_function(lambda x, y: mms.u(x, y, mms.t_n))
    p_n.set_values_from_function(lambda x, y: float(mms.p(x, y, mms.t_n)))
    phi_n.set_values_from_function(lambda x, y: float(mms.phi(x, y, mms.t_n)))
    alpha_n.set_values_from_function(lambda x, y: float(mms.alpha(x, y, mms.t_n)))
    if alpha_ch_enabled:
        if getattr(mms, "mu_alpha", None) is None:
            raise ValueError("alpha_ch_* enabled but MMS does not provide mu_alpha(x,y,t).")
        mu_alpha_n.set_values_from_function(lambda x, y: float(mms.mu_alpha(x, y, mms.t_n)))
        mu_alpha_k.nodal_values[:] = mu_alpha_n.nodal_values
    if alpha_cahn_conservative:
        lambda_alpha_n.nodal_values.fill(float(getattr(mms, "lambda_alpha_n", 0.0)))
        lambda_alpha_k.nodal_values[:] = lambda_alpha_n.nodal_values
    S_n.set_values_from_function(lambda x, y: float(mms.S(x, y, mms.t_n)))
    X_n.set_values_from_function(lambda x, y: float(mms.X(x, y, mms.t_n)))

    dt_c = Constant(float(dt_val))

    f_v = Analytic(lambda x, y: mms.f_v(x, y), degree=6)
    f_u = Analytic(lambda x, y: mms.f_u(x, y), degree=6)
    s_v = Analytic(lambda x, y: mms.s_v(x, y), degree=6)
    f_phi = Analytic(lambda x, y: mms.f_phi(x, y), degree=6)
    f_alpha = Analytic(lambda x, y: mms.f_alpha(x, y), degree=6)
    f_S = Analytic(lambda x, y: mms.f_S(x, y), degree=6)
    D_det_prev = Analytic(lambda x, y: mms.D_det_prev(x, y), degree=6)
    f_X = Analytic(lambda x, y: mms.f_X(x, y), degree=6)

    forms = build_biofilm_one_domain_forms(
        v_k=v_k,
        p_k=p_k,
        u_k=u_k,
        phi_k=phi_k,
        alpha_k=alpha_k,
        mu_alpha_k=mu_alpha_k,
        lambda_alpha_k=lambda_alpha_k,
        S_k=S_k,
        X_k=X_k,
        v_n=v_n,
        p_n=p_n,
        u_n=u_n,
        phi_n=phi_n,
        alpha_n=alpha_n,
        mu_alpha_n=mu_alpha_n,
        lambda_alpha_n=lambda_alpha_n,
        S_n=S_n,
        X_n=X_n,
        dv=dv,
        dp=dp,
        du=du,
        dphi=dphi,
        dalpha=dalpha,
        dmu_alpha=dmu_alpha,
        dlambda_alpha=dlambda_alpha,
        dS=dS,
        dX=dX,
        v_test=v_test,
        q_test=q_test,
        u_test=u_test,
        phi_test=phi_test,
        alpha_test=alpha_test,
        mu_alpha_test=mu_alpha_test,
        lambda_alpha_test=lambda_alpha_test,
        S_test=S_test,
        X_test=X_test,
        dx=dx(metadata={"q": int(qdeg)}),
        dt=dt_c,
        theta=float(theta),
        rho_f=Constant(1.0),
        mu_f=Constant(1.0e-2),
        kappa_inv=Constant(10.0),
        mu_s=Constant(1.0),
        lambda_s=Constant(1.0),
        D_phi=0.1,
        gamma_phi=1.0,
        D_alpha=0.0 if alpha_ch_enabled else 0.1,
        alpha_ch_M=float(getattr(mms, "alpha_ch_M", 0.0) or 0.0),
        alpha_ch_gamma=float(getattr(mms, "alpha_ch_gamma", 0.0) or 0.0),
        alpha_ch_eps=float(getattr(mms, "alpha_ch_eps", 1.0) or 1.0),
        alpha_ch_mobility=str(getattr(mms, "alpha_ch_mobility", "constant")),
        alpha_cahn_M=float(getattr(mms, "alpha_cahn_M", 0.0) or 0.0),
        alpha_cahn_gamma=float(getattr(mms, "alpha_cahn_gamma", 0.0) or 0.0),
        alpha_cahn_eps=float(getattr(mms, "alpha_cahn_eps", 1.0) or 1.0),
        alpha_cahn_conservative=alpha_cahn_conservative,
        alpha_cahn_mobility=str(getattr(mms, "alpha_cahn_mobility", "constant")),
        D_S=0.1,
        D_X=0.1,
        mu_max=0.4,
        K_S=0.3,
        k_g=0.5,
        k_d=0.1,
        Y=0.8,
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
                BoundaryCondition("u_x", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.u(x, y, t)[..., 0])),
                BoundaryCondition("u_y", "dirichlet", tag, _as_float_time(lambda x, y, t: mms.u(x, y, t)[..., 1])),
                BoundaryCondition("phi", "dirichlet", tag, _as_float_time(mms.phi)),
                BoundaryCondition("alpha", "dirichlet", tag, _as_float_time(mms.alpha)),
                *( [BoundaryCondition("mu_alpha", "dirichlet", tag, _as_float_time(mms.mu_alpha))] if alpha_ch_enabled else []),
                BoundaryCondition("S", "dirichlet", tag, _as_float_time(mms.S)),
                BoundaryCondition("X", "dirichlet", tag, _as_float_time(mms.X)),
            ]
        )
    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, (lambda x, y: 0.0)) for b in bcs]

    solver = NewtonSolver(
        forms.residual_form,
        forms.jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=float(newton_tol), max_newton_iter=int(max_it)),
        quad_order=int(qdeg),
        backend="cpp",
    )

    solver.solve_time_interval(
        functions=[
            v_k,
            p_k,
            u_k,
            phi_k,
            alpha_k,
            *( [mu_alpha_k] if alpha_ch_enabled else []),
            *( [lambda_alpha_k] if alpha_cahn_conservative else []),
            S_k,
            X_k,
        ],
        prev_functions=[
            v_n,
            p_n,
            u_n,
            phi_n,
            alpha_n,
            *( [mu_alpha_n] if alpha_ch_enabled else []),
            *( [lambda_alpha_n] if alpha_cahn_conservative else []),
            S_n,
            X_n,
        ],
        aux_functions={"dt": dt_c},
        time_params=TimeStepperParameters(dt=float(dt_val), final_time=float(dt_val), max_steps=1, theta=float(theta)),
    )

    t_err = float(mms.t_k)
    err_v = dh.l2_error(
        v_k,
        exact={"v_x": lambda x, y: mms.v(x, y, t_err)[..., 0], "v_y": lambda x, y: mms.v(x, y, t_err)[..., 1]},
        fields=["v_x", "v_y"],
        quad_order=int(qerr),
        relative=False,
    )
    err_p = dh.l2_error(p_k, exact={"p": lambda x, y: mms.p(x, y, t_err)}, fields=["p"], quad_order=int(qerr), relative=False)
    err_alpha = dh.l2_error(alpha_k, exact={"alpha": lambda x, y: mms.alpha(x, y, t_err)}, fields=["alpha"], quad_order=int(qerr), relative=False)
    err_mu_alpha = None
    if alpha_ch_enabled:
        err_mu_alpha = dh.l2_error(
            mu_alpha_k,
            exact={"mu_alpha": lambda x, y: mms.mu_alpha(x, y, t_err)},
            fields=["mu_alpha"],
            quad_order=int(qerr),
            relative=False,
        )

    return {
        "h": 1.0 / float(nx),
        "err_v": float(err_v),
        "err_p": float(err_p),
        "err_alpha": float(err_alpha),
        "err_mu_alpha": float(err_mu_alpha) if err_mu_alpha is not None else None,
    }


def test_biofilm_one_domain_mms_trig_convergence_cpp():
    # Keep this convergence check lightweight: it is a regression guard, not a
    # full accuracy study. The full biofilm one-domain solve has many fields and
    # can dominate test runtimes.
    #
    # To run a heavier study locally, set `PYCUTFEM_BIOFILM_MMS_NX_LIST`, e.g.:
    #   PYCUTFEM_BIOFILM_MMS_NX_LIST=4,8
    dt_val = 0.05
    theta = 1.0
    k_det = 0.2
    qdeg = int(os.environ.get("PYCUTFEM_BIOFILM_MMS_QDEG", "6") or "6")
    qerr = int(os.environ.get("PYCUTFEM_BIOFILM_MMS_QERR", str(qdeg)) or str(qdeg))
    newton_tol = float(os.environ.get("PYCUTFEM_BIOFILM_MMS_NEWTON_TOL", "1.0e-8") or "1.0e-8")
    max_it = int(os.environ.get("PYCUTFEM_BIOFILM_MMS_NEWTON_MAX_IT", "20") or "20")

    nx_spec = str(os.environ.get("PYCUTFEM_BIOFILM_MMS_NX_LIST", "")).strip()
    if nx_spec:
        nx_list = [int(x.strip()) for x in nx_spec.split(",") if x.strip()]
    else:
        nx_list = [3, 6]
    if len(nx_list) < 2:
        raise ValueError("PYCUTFEM_BIOFILM_MMS_NX_LIST must contain at least two mesh sizes (e.g. '3,6').")

    mms = build_biofilm_one_domain_mms_trig_step(dt_val=float(dt_val), theta=float(theta), k_det=float(k_det))

    errs = [
        _solve_one(
            nx=int(nx),
            qdeg=qdeg,
            qerr=qerr,
            dt_val=dt_val,
            theta=theta,
            newton_tol=newton_tol,
            max_it=max_it,
            k_det=k_det,
            mms=mms,
        )
        for nx in nx_list[:2]
    ]
    coarse, fine = errs[0], errs[1]
    assert fine["err_v"] < coarse["err_v"]
    assert fine["err_p"] < coarse["err_p"]
    assert fine["err_alpha"] < coarse["err_alpha"]

    # Expect clear improvement under refinement. Keep thresholds modest to avoid
    # flakiness on coarse meshes while still catching regressions (EOC ~ 0).
    eoc_v = _eoc(coarse["h"], fine["h"], coarse["err_v"], fine["err_v"])
    eoc_p = _eoc(coarse["h"], fine["h"], coarse["err_p"], fine["err_p"])
    eoc_alpha = _eoc(coarse["h"], fine["h"], coarse["err_alpha"], fine["err_alpha"])

    assert eoc_v > 1.0
    assert eoc_p > 0.7
    assert eoc_alpha > 1.0


def test_biofilm_one_domain_mms_trig_convergence_cpp_conservative_alpha_cahn():
    # Lightweight convergence regression for the conservative Allen–Cahn extension:
    # ensure α convergence under refinement when λ_α is included as a global unknown.
    dt_val = 0.05
    theta = 1.0
    k_det = 0.2
    qdeg = 6
    qerr = 6
    newton_tol = 1.0e-7
    max_it = 15

    mms = build_biofilm_one_domain_mms_trig_step(
        dt_val=float(dt_val),
        theta=float(theta),
        k_det=float(k_det),
        alpha_cahn_M=0.2,
        alpha_cahn_gamma=1.0,
        alpha_cahn_eps=0.1,
        alpha_cahn_conservative=True,
        alpha_cahn_mobility="constant",
        alpha_wave=2,
    )

    errs = [
        _solve_one(nx=int(nx), qdeg=qdeg, qerr=qerr, dt_val=dt_val, theta=theta, newton_tol=newton_tol, max_it=max_it, k_det=k_det, mms=mms)
        for nx in (2, 4)
    ]
    coarse, fine = errs[0], errs[1]
    assert fine["err_alpha"] < coarse["err_alpha"]

    eoc_alpha = _eoc(coarse["h"], fine["h"], coarse["err_alpha"], fine["err_alpha"])
    assert eoc_alpha > 1.0


## NOTE: Cahn–Hilliard MMS convergence is covered by a dedicated two-field CH MMS test
## (alpha, mu_alpha) to keep the test suite runtime manageable.
