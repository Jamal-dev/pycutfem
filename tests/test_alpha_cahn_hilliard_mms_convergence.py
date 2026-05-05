import math

import numpy as np
import sympy as sp

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction, grad, inner
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import dx
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


def _build_ch_mms(*, dt_val: float, M0: float, gamma: float, eps: float, mobility: str):
    dt = float(dt_val)
    if not (dt > 0.0):
        raise ValueError("dt_val must be positive.")

    x, y, t = sp.symbols("x y t", real=True)
    s = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    alpha_expr = sp.Rational(1, 2) + sp.Rational(1, 5) * s * (1.0 + sp.Rational(1, 10) * sp.sin(t))

    alpha_n_expr = alpha_expr.subs({t: 0.0})
    alpha_k_expr = alpha_expr.subs({t: dt})

    def lap(f):
        return sp.diff(f, x, 2) + sp.diff(f, y, 2)

    def grad2(f):
        return sp.Matrix([sp.diff(f, x), sp.diff(f, y)])

    # W'(a) for W(a)=a^2(1-a)^2 is 2a(1-a)(1-2a).
    def Wp(a):
        return 2.0 * a * (1.0 - a) * (1.0 - 2.0 * a)

    mu_expr = float(gamma) * ((-float(eps)) * lap(alpha_expr) + (Wp(alpha_expr) / float(eps)))
    mu_n_expr = mu_expr.subs({t: 0.0})
    mu_k_expr = mu_expr.subs({t: dt})

    mob_key = str(mobility).strip().lower()
    if mob_key in {"constant", "const"}:
        M_k_expr = float(M0)
    elif mob_key in {"degenerate", "deg", "alpha(1-alpha)", "alpha*(1-alpha)", "a(1-a)"}:
        M_k_expr = float(M0) * alpha_k_expr * (1.0 - alpha_k_expr)
    else:
        raise ValueError(f"Unknown mobility={mobility!r}. Use 'constant' or 'degenerate'.")

    # f_alpha for BE: (alpha_k-alpha_n)/dt - div(M grad(mu_k))
    flux_k = M_k_expr * grad2(mu_k_expr)
    div_flux_k = sp.diff(flux_k[0], x) + sp.diff(flux_k[1], y)
    f_alpha_expr = (alpha_k_expr - alpha_n_expr) / dt - div_flux_k

    alpha_fn = sp.lambdify((x, y, t), alpha_expr, "numpy")
    mu_fn = sp.lambdify((x, y, t), mu_expr, "numpy")
    f_alpha_fn = sp.lambdify((x, y), f_alpha_expr, "numpy")

    return {
        "alpha": lambda xv, yv, tv: np.asarray(alpha_fn(xv, yv, tv), dtype=float),
        "mu": lambda xv, yv, tv: np.asarray(mu_fn(xv, yv, tv), dtype=float),
        "f_alpha": lambda xv, yv: np.asarray(f_alpha_fn(xv, yv), dtype=float),
    }


def _solve_one(*, nx: int, qdeg: int, qerr: int, dt_val: float, M0: float, gamma: float, eps: float, mobility: str):
    mms = _build_ch_mms(dt_val=dt_val, M0=M0, gamma=gamma, eps=eps, mobility=mobility)

    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=int(nx), ny=int(nx), poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    _tag_unit_square_boundaries(mesh)

    me = MixedElement(mesh, field_specs={"alpha": 1, "mu_alpha": 1})
    dh = DofHandler(me, method="cg")

    dalpha = TrialFunction("alpha", dof_handler=dh)
    dmu = TrialFunction("mu_alpha", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    mu_test = TestFunction("mu_alpha", dof_handler=dh)

    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    mu_k = Function("mu_k", "mu_alpha", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    mu_n = Function("mu_n", "mu_alpha", dof_handler=dh)

    alpha_n.set_values_from_function(lambda x, y: float(mms["alpha"](x, y, 0.0)))
    mu_n.set_values_from_function(lambda x, y: float(mms["mu"](x, y, 0.0)))
    alpha_k.nodal_values[:] = alpha_n.nodal_values
    mu_k.nodal_values[:] = mu_n.nodal_values

    dt_c = Constant(float(dt_val))
    inv_dt = Constant(1.0 / float(dt_val))

    M0_c = Constant(float(M0))
    gamma_c = Constant(float(gamma))
    eps_c = Constant(float(eps))

    def _one_minus(expr):
        return (-expr) + Constant(1.0)

    mob_key = str(mobility).strip().lower()
    if mob_key in {"constant", "const"}:
        M_k = M0_c
        dM = Constant(0.0) * dalpha
    else:
        mob = alpha_k * _one_minus(alpha_k)
        mob_prime = _one_minus(Constant(2.0) * alpha_k)
        M_k = M0_c * mob
        dM = M0_c * mob_prime * dalpha

    # Double-well derivatives for W(a)=a^2(1-a)^2.
    Wp_k = Constant(2.0) * alpha_k * _one_minus(alpha_k) * _one_minus(Constant(2.0) * alpha_k)
    Wpp_k = (-Constant(12.0) * alpha_k) + (Constant(12.0) * (alpha_k * alpha_k)) + Constant(2.0)

    f_alpha = Analytic(lambda x, y: mms["f_alpha"](x, y), degree=8)

    r_alpha = alpha_test * ((alpha_k - alpha_n) * inv_dt) * dx(metadata={"q": int(qdeg)})
    r_alpha += inner(M_k * grad(mu_k), grad(alpha_test)) * dx(metadata={"q": int(qdeg)})
    r_alpha += -alpha_test * f_alpha * dx(metadata={"q": int(qdeg)})

    r_mu = mu_test * mu_k * dx(metadata={"q": int(qdeg)})
    r_mu += -(gamma_c * eps_c) * inner(grad(alpha_k), grad(mu_test)) * dx(metadata={"q": int(qdeg)})
    r_mu += -mu_test * ((gamma_c / eps_c) * Wp_k) * dx(metadata={"q": int(qdeg)})

    residual_form = r_alpha + r_mu

    a_alpha = alpha_test * (dalpha * inv_dt) * dx(metadata={"q": int(qdeg)})
    a_alpha += (dM * inner(grad(mu_k), grad(alpha_test))) * dx(metadata={"q": int(qdeg)})
    a_alpha += inner(M_k * grad(dmu), grad(alpha_test)) * dx(metadata={"q": int(qdeg)})

    a_mu = mu_test * dmu * dx(metadata={"q": int(qdeg)})
    a_mu += -(gamma_c * eps_c) * inner(grad(dalpha), grad(mu_test)) * dx(metadata={"q": int(qdeg)})
    a_mu += -mu_test * ((gamma_c / eps_c) * Wpp_k * dalpha) * dx(metadata={"q": int(qdeg)})

    jacobian_form = a_alpha + a_mu

    bcs = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                BoundaryCondition("alpha", "dirichlet", tag, _as_float_time(mms["alpha"])),
                BoundaryCondition("mu_alpha", "dirichlet", tag, _as_float_time(mms["mu"])),
            ]
        )
    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, (lambda x, y: 0.0)) for b in bcs]

    solver = NewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=1.0e-10, max_newton_iter=20),
        quad_order=int(qdeg),
        backend="cpp",
    )

    solver.solve_time_interval(
        functions=[alpha_k, mu_k],
        prev_functions=[alpha_n, mu_n],
        aux_functions={"dt": dt_c},
        time_params=TimeStepperParameters(dt=float(dt_val), final_time=float(dt_val), max_steps=1, theta=1.0),
    )

    t_err = float(dt_val)
    err_alpha = dh.l2_error(alpha_k, exact={"alpha": lambda x, y: mms["alpha"](x, y, t_err)}, fields=["alpha"], quad_order=int(qerr), relative=False)
    err_mu = dh.l2_error(mu_k, exact={"mu_alpha": lambda x, y: mms["mu"](x, y, t_err)}, fields=["mu_alpha"], quad_order=int(qerr), relative=False)

    return {"h": 1.0 / float(nx), "err_alpha": float(err_alpha), "err_mu": float(err_mu)}


def test_alpha_cahn_hilliard_mms_convergence_cpp():
    dt_val = 0.05
    M0 = 0.4
    gamma = 1.0
    eps = 0.1
    mobility = "degenerate"
    qdeg = 6
    qerr = 6

    errs = [
        _solve_one(nx=int(nx), qdeg=qdeg, qerr=qerr, dt_val=dt_val, M0=M0, gamma=gamma, eps=eps, mobility=mobility)
        for nx in (6, 12)
    ]
    coarse, fine = errs[0], errs[1]
    assert fine["err_alpha"] < coarse["err_alpha"]
    assert fine["err_mu"] < coarse["err_mu"]

    eoc_alpha = _eoc(coarse["h"], fine["h"], coarse["err_alpha"], fine["err_alpha"])
    eoc_mu = _eoc(coarse["h"], fine["h"], coarse["err_mu"], fine["err_mu"])

    assert eoc_alpha > 1.0
    assert eoc_mu > 1.0
