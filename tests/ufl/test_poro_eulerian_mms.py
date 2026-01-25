import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pytest
import sympy as sp

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import Constant, Function, Identity, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction, dot
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dS, dx
from pycutfem.utils.fpi_poro_eulerian import jacobian_poro, residual_poro
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


@dataclass(frozen=True)
class _PoroMMS:
    # exact fields at t_{n+1} (we use a single BE step with t_n=0, t_{n+1}=dt)
    v: callable  # (x,y)->(...,2)
    u: callable  # (x,y)->(...,2)
    p: callable  # (x,y)->(...)
    # forcing terms in strong form, mapped to our weak form
    f_mass: callable  # scalar
    f_v: callable  # (x,y)->(...,2)
    f_u: callable  # (x,y)->(...,2)
    # Neumann traction for the skeleton on selected boundaries:
    #   t = (sigma(u) + phi p I) n
    t_right: callable  # (x,y)->(...,2)
    t_top: callable  # (x,y)->(...,2)
    # scalar component callables for Dirichlet BCs
    v_x: callable
    v_y: callable
    u_x: callable
    u_y: callable
    p_s: callable


def _kinv_ref_matrix(case: str) -> np.ndarray:
    case = str(case).strip().lower()
    if case in {"iso", "identity", "i"}:
        return np.eye(2, dtype=float)
    if case in {"aniso", "anisotropic", "a"}:
        return np.array([[2.0, 0.3], [0.1, 1.5]], dtype=float)
    raise ValueError(f"Unknown K_inv case {case!r}")


@lru_cache(maxsize=None)
def _build_poro_mms(*, kinv_case: str):
    """
    Build a fully-Eulerian porous MMS using SymPy and lambdify it to NumPy callables.

    We use a single backward-Euler step (theta=1):
      t_n = 0,  t_{n+1} = dt.
    """
    kinv_case = str(kinv_case).strip().lower()
    x, y = sp.symbols("x y", real=True)
    pi = sp.pi

    # ---- numerical parameters used by the MMS (kept small for robust Newton) ----
    dt_val = 0.1
    A = 0.05
    B = 0.03
    P0 = 0.2
    phi = 0.6
    rho_f = 0.9
    mu_f = 1.1
    rho_s = 1.0
    c_nh = 0.7
    beta_nh = 0.0

    dt = sp.Float(dt_val)

    # ---- exact solution at t_{n+1}=dt (smooth, small) ----
    # u ~ O(dt^2)  => v_s = (u^{n+1}-u^n)/dt ~ O(dt) (small)
    u1 = sp.Float(A) * (dt**2) * sp.sin(pi * x) * sp.sin(pi * y)
    u2 = sp.Float(A) * (dt**2) * sp.cos(pi * x) * sp.sin(pi * y)
    u = sp.Matrix([u1, u2])

    # v ~ O(dt), p ~ O(dt)
    v1 = sp.Float(B) * dt * sp.cos(pi * x) * sp.sin(pi * y)
    v2 = -sp.Float(B) * dt * sp.sin(pi * x) * sp.cos(pi * y)
    v = sp.Matrix([v1, v2])

    # Make p vary on the boundary as well (avoid a constant Dirichlet value),
    # which otherwise leaves the mean-pressure mode weakly constrained in this
    # mixed formulation and can pollute MMS convergence across backends.
    p = sp.Float(P0) * dt * (x + y + sp.sin(pi * x) * sp.sin(pi * y))

    def _grad_vec(a: sp.Matrix) -> sp.Matrix:
        return sp.Matrix([[sp.diff(a[0], x), sp.diff(a[0], y)], [sp.diff(a[1], x), sp.diff(a[1], y)]])

    def _div_vec(a: sp.Matrix) -> sp.Expr:
        return sp.diff(a[0], x) + sp.diff(a[1], y)

    # ---- Eulerian kinematics (reference map) ----
    I2 = sp.eye(2)
    grad_u = _grad_vec(u)
    F = (I2 - grad_u).inv()
    J = sp.det(F)
    Finv = I2 - grad_u  # exact: inv(F) = I - grad(u)

    K_inv = sp.Matrix(_kinv_ref_matrix(kinv_case))
    kinv = J * Finv.T * K_inv * Finv

    # ---- discrete BE time derivatives ----
    vdot = v / dt
    v_s = u / dt

    # ---- strong residuals (mapped to weak form via body force + traction) ----
    mix = sp.Float(phi) * v + sp.Float(1.0 - phi) * v_s
    f_mass = _div_vec(mix)

    grad_v = _grad_vec(v)
    conv = -sp.Float(rho_f) * (grad_v * v_s)
    drag = sp.Float(mu_f) * (sp.Float(phi) ** 2) * (kinv * (v - v_s))
    grad_p = sp.Matrix([sp.diff(p, x), sp.diff(p, y)])
    f_v = sp.Float(rho_f) * vdot + conv + drag + grad_p

    # Neo-Hookean skeleton Cauchy stress (beta=0 => a=1)
    Bmat = F * F.T
    a = J ** (-2.0 * sp.Float(beta_nh))
    sigma = (2.0 * sp.Float(c_nh) / J) * (Bmat - a * I2)
    div_sigma = sp.Matrix(
        [
            sp.diff(sigma[0, 0], x) + sp.diff(sigma[0, 1], y),
            sp.diff(sigma[1, 0], x) + sp.diff(sigma[1, 1], y),
        ]
    )

    acc_local = v_s / dt
    adv = _grad_vec(v_s) * v_s
    acc = acc_local + adv
    f_u = sp.Float(rho_s) * acc - div_sigma - drag - sp.Float(phi) * grad_p

    # Neumann traction for the combined (sigma + phi p I) stress on Γ_N
    t = sigma + sp.Float(phi) * p * I2
    t_right = t * sp.Matrix([1.0, 0.0])
    t_top = t * sp.Matrix([0.0, 1.0])

    # ---- lambdify ----
    v1_fun = sp.lambdify((x, y), v1, "numpy")
    v2_fun = sp.lambdify((x, y), v2, "numpy")
    u1_fun = sp.lambdify((x, y), u1, "numpy")
    u2_fun = sp.lambdify((x, y), u2, "numpy")
    p_fun = sp.lambdify((x, y), p, "numpy")

    f_mass_fun = sp.lambdify((x, y), f_mass, "numpy")
    f_v0 = sp.lambdify((x, y), f_v[0], "numpy")
    f_v1 = sp.lambdify((x, y), f_v[1], "numpy")
    f_u0 = sp.lambdify((x, y), f_u[0], "numpy")
    f_u1 = sp.lambdify((x, y), f_u[1], "numpy")
    t_r0 = sp.lambdify((x, y), t_right[0], "numpy")
    t_r1 = sp.lambdify((x, y), t_right[1], "numpy")
    t_t0 = sp.lambdify((x, y), t_top[0], "numpy")
    t_t1 = sp.lambdify((x, y), t_top[1], "numpy")

    def _vec2(f0, f1):
        def cb(xv, yv):
            return np.stack((f0(xv, yv), f1(xv, yv)), axis=-1)

        return cb

    return _PoroMMS(
        v=_vec2(v1_fun, v2_fun),
        u=_vec2(u1_fun, u2_fun),
        p=p_fun,
        f_mass=f_mass_fun,
        f_v=_vec2(f_v0, f_v1),
        f_u=_vec2(f_u0, f_u1),
        t_right=_vec2(t_r0, t_r1),
        t_top=_vec2(t_t0, t_t1),
        v_x=lambda xx, yy: float(v1_fun(xx, yy)),
        v_y=lambda xx, yy: float(v2_fun(xx, yy)),
        u_x=lambda xx, yy: float(u1_fun(xx, yy)),
        u_y=lambda xx, yy: float(u2_fun(xx, yy)),
        p_s=lambda xx, yy: float(p_fun(xx, yy)),
    )


def _solve_one_step_poro_mms(
    *,
    nx: int,
    backend: str,
    deg_v: int,
    deg_u: int,
    deg_p: int,
    kinv_case: str,
):
    mms = _build_poro_mms(kinv_case=kinv_case)
    dt_val = 0.1
    max_deg = int(max(deg_v, deg_u, deg_p))
    # Nonlinear kinematics (det/inv) are sensitive to quadrature. Compiled
    # backends can afford a higher quadrature; keep python backend cheaper.
    if backend == "python":
        qdeg = max(4, 2 * max_deg + 2)
    else:
        qdeg = max(6, 2 * max_deg + 4)

    poly_order = int(max(deg_v, deg_u, deg_p))
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=int(nx), ny=int(nx), poly_order=poly_order, offset=(0.0, 0.0))
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )
    _tag_unit_square_boundaries(mesh)

    me = MixedElement(mesh, field_specs={"v_x": deg_v, "v_y": deg_v, "u_x": deg_u, "u_y": deg_u, "p": deg_p})
    dh = DofHandler(me, method="cg")

    Vv = FunctionSpace(name="v", field_names=["v_x", "v_y"], dim=1)
    Vu = FunctionSpace(name="u", field_names=["u_x", "u_y"], dim=1)

    dv = VectorTrialFunction(space=Vv, dof_handler=dh)
    du = VectorTrialFunction(space=Vu, dof_handler=dh)
    dp = TrialFunction(name="dp", field_name="p", dof_handler=dh)

    w = VectorTestFunction(space=Vv, dof_handler=dh)
    eta = VectorTestFunction(space=Vu, dof_handler=dh)
    q = TestFunction(name="q", field_name="p", dof_handler=dh)

    v_k = VectorFunction(name="v_k", field_names=["v_x", "v_y"], dof_handler=dh)
    u_k = VectorFunction(name="u_k", field_names=["u_x", "u_y"], dof_handler=dh)
    p_k = Function(name="p_k", field_name="p", dof_handler=dh)
    v_n = VectorFunction(name="v_n", field_names=["v_x", "v_y"], dof_handler=dh)
    u_n = VectorFunction(name="u_n", field_names=["u_x", "u_y"], dof_handler=dh)
    p_n = Function(name="p_n", field_name="p", dof_handler=dh)

    for f in (v_k, u_k, p_k, v_n, u_n, p_n):
        f.nodal_values.fill(0.0)

    dx_p = dx(metadata={"q": qdeg})

    dt = Constant(dt_val)
    dt._jit_name = "dt"
    theta = Constant(1.0)
    theta._jit_name = "theta"

    # physical/material parameters (match the MMS builder above)
    rho_f = Constant(0.9)
    rho_f._jit_name = "rho_f"
    mu_f = Constant(1.1)
    mu_f._jit_name = "mu_f"
    rho_s0_tilde = Constant(1.0)
    rho_s0_tilde._jit_name = "rho_s0_tilde"
    phi = Constant(0.6)
    phi._jit_name = "phi"
    if str(kinv_case).strip().lower() in {"iso", "identity", "i"}:
        K_inv = Identity(2)
    else:
        K_inv = Constant(_kinv_ref_matrix(kinv_case).tolist(), dim=2)
        K_inv._jit_name = "K_inv"
    c_nh = Constant(0.7)
    c_nh._jit_name = "c_nh"
    beta_nh = Constant(0.0)
    beta_nh._jit_name = "beta_nh"

    r = residual_poro(
        v_k,
        u_k,
        p_k,
        v_n,
        u_n,
        p_n,
        q,
        w,
        eta,
        rho_f=rho_f,
        mu_f=mu_f,
        rho_s0_tilde=rho_s0_tilde,
        phi=phi,
        K_inv=K_inv,
        c_nh=c_nh,
        beta_nh=beta_nh,
        dt=dt,
        theta=theta,
        dx_p=dx_p,
    )
    a = jacobian_poro(
        v_k,
        u_k,
        p_k,
        u_n,
        dv,
        du,
        dp,
        q,
        w,
        eta,
        rho_f=rho_f,
        mu_f=mu_f,
        rho_s0_tilde=rho_s0_tilde,
        phi=phi,
        K_inv=K_inv,
        c_nh=c_nh,
        beta_nh=beta_nh,
        dt=dt,
        theta=theta,
        dx_p=dx_p,
    )

    # body forces (volume)
    ana_deg = max(8, qdeg)
    f_mass = Analytic(mms.f_mass, degree=ana_deg)
    f_v = Analytic(mms.f_v, degree=ana_deg)
    f_u = Analytic(mms.f_u, degree=ana_deg)
    r += -(f_mass * q) * dx_p
    r += -(dot(f_v, w)) * dx_p
    r += -(dot(f_u, eta)) * dx_p

    # Dirichlet BCs:
    # - velocity and pressure on the whole boundary (simplifies pressure-gradient boundary terms)
    # - displacement on the whole boundary (keeps the MMS solve robust across backends)
    bcs = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                BoundaryCondition("v_x", "dirichlet", tag, mms.v_x),
                BoundaryCondition("v_y", "dirichlet", tag, mms.v_y),
                BoundaryCondition("p", "dirichlet", tag, mms.p_s),
                BoundaryCondition("u_x", "dirichlet", tag, mms.u_x),
                BoundaryCondition("u_y", "dirichlet", tag, mms.u_y),
            ]
        )
    bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x, y: 0.0) for bc in bcs]

    solver = NewtonSolver(
        r,
        a,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        # Keep the MMS test reasonably fast on the pure-python backend: disable line-search
        # (Newton converges in a few iterations for this small-amplitude manufactured state).
        newton_params=NewtonParameters(
            newton_tol=1.0e-10,
            max_newton_iter=12,
            # Line-search improves robustness on compiled backends; keep it off for python for speed.
            line_search=(backend != "python"),
        ),
        quad_order=qdeg,
        backend=backend,
    )

    solver.solve_time_interval(
        functions=[v_k, u_k, p_k],
        prev_functions=[v_n, u_n, p_n],
        aux_functions={"dt": dt},
        time_params=TimeStepperParameters(dt=dt_val, max_steps=1, final_time=dt_val, theta=1.0),
    )

    err_v = dh.l2_error(
        v_k,
        exact={"v_x": lambda x, y: np.asarray(mms.v(x, y))[..., 0], "v_y": lambda x, y: np.asarray(mms.v(x, y))[..., 1]},
        quad_order=qdeg,
        relative=False,
    )
    err_u = dh.l2_error(
        u_k,
        exact={"u_x": lambda x, y: np.asarray(mms.u(x, y))[..., 0], "u_y": lambda x, y: np.asarray(mms.u(x, y))[..., 1]},
        quad_order=qdeg,
        relative=False,
    )
    err_p = dh.l2_error(
        p_k,
        exact={"p": lambda x, y: np.asarray(mms.p(x, y))},
        quad_order=qdeg,
        relative=False,
    )

    return {"v": float(err_v), "u": float(err_u), "p": float(err_p)}


def _print_convergence_table(*, backend: str, degs: tuple[int, int, int], kinv_case: str, nx_list: list[int], errs: list[dict[str, float]]):
    def _eoc(prev: float, curr: float) -> float:
        return math.log(prev / curr, 2.0)

    deg_v, deg_u, deg_p = degs
    title = f"Porous MMS convergence | backend={backend} | deg(v,u,p)=({deg_v},{deg_u},{deg_p}) | K_inv={kinv_case}"
    print("\n" + title)
    print("-" * len(title))
    print(f"{'nx':>4} {'h':>8} | {'err_v':>12} {'eoc_v':>6} | {'err_u':>12} {'eoc_u':>6} | {'err_p':>12} {'eoc_p':>6}")
    for i, (nx, e) in enumerate(zip(nx_list, errs, strict=True)):
        h = 1.0 / float(nx)
        if i == 0:
            eoc_v = eoc_u = eoc_p = float("nan")
        else:
            eoc_v = _eoc(errs[i - 1]["v"], e["v"])
            eoc_u = _eoc(errs[i - 1]["u"], e["u"])
            eoc_p = _eoc(errs[i - 1]["p"], e["p"])
        eoc_fmt = lambda v: ("   - " if not np.isfinite(v) else f"{v:6.2f}")
        print(
            f"{nx:4d} {h:8.3e} | {e['v']:12.5e} {eoc_fmt(eoc_v)} | {e['u']:12.5e} {eoc_fmt(eoc_u)} | {e['p']:12.5e} {eoc_fmt(eoc_p)}"
        )


def _selected_backends(*, default: str = "cpp") -> list[str]:
    """
    Backends to run in MMS tests.

    - `PYCUTFEM_TEST_BACKENDS=python,jit,cpp` (preferred)
    - `BACKEND=python|jit|cpp|all|python,jit` (fallback)
    """
    import os

    spec = (os.environ.get("PYCUTFEM_TEST_BACKENDS") or os.environ.get("BACKEND") or default).strip()
    if not spec:
        return [default]
    if spec.lower() == "all":
        return ["python", "jit", "cpp"]
    backends = [b.strip() for b in spec.split(",") if b.strip()]
    valid = {"python", "jit", "cpp"}
    unknown = [b for b in backends if b not in valid]
    if unknown:
        raise ValueError(f"Unknown backend(s) {unknown}; valid={sorted(valid)}")
    return backends


@pytest.mark.parametrize("kinv_case", ["iso", "aniso"])
def test_poro_eulerian_mms_convergence(kinv_case):
    # Use an inf-sup stable discretization for robust pressure convergence:
    # (v,u) as Q2 and p as Q1 (Taylor–Hood-like).
    #
    # Equal-order (Q1/Q1 or Q2/Q2) is useful for experimentation, but it can
    # exhibit pressure instabilities (especially for anisotropic K^{-1}).
    deg_v, deg_u, deg_p = 2, 2, 1

    for backend in _selected_backends(default="jit"):
        # Keep the pure-python backend reasonably fast.
        nx_list = [2, 4] if backend == "python" else [4, 8]

        errs = [
            _solve_one_step_poro_mms(
                nx=nx,
                backend=backend,
                deg_v=deg_v,
                deg_u=deg_u,
                deg_p=deg_p,
                kinv_case=kinv_case,
            )
            for nx in nx_list
        ]

        _print_convergence_table(
            backend=backend,
            degs=(deg_v, deg_u, deg_p),
            kinv_case=kinv_case,
            nx_list=list(nx_list),
            errs=errs,
        )

        # Basic correctness: errors are finite and decrease under refinement.
        for key in ("v", "u", "p"):
            series = [float(e[key]) for e in errs]
            assert all(np.isfinite(series))
            assert all(val > 1.0e-14 for val in series)
            assert series[-1] < 0.95 * series[0]


def _run_equal_order_demo() -> bool:
    import os

    return os.environ.get("PYCUTFEM_PORO_MMS_EQUAL_ORDER", "").strip().lower() in {"1", "true", "yes"}


@pytest.mark.skipif(not _run_equal_order_demo(), reason="Set PYCUTFEM_PORO_MMS_EQUAL_ORDER=1 to run equal-order convergence demo.")
@pytest.mark.parametrize("kinv_case", ["iso", "aniso"])
def test_poro_eulerian_mms_equal_order_convergence_demo(kinv_case):
    """
    Optional demo: equal-order convergence behaviour (Q1/Q1/Q1 vs Q2/Q2/Q2).

    This is not asserted because pressure can be unstable for equal-order pairs.
    It is useful when debugging element choices or stabilization additions.
    """
    cases = [(1, 1, 1), (2, 2, 2)]
    for backend in _selected_backends(default="jit"):
        nx_list = [2, 4] if backend == "python" else [4, 8]
        for deg_v, deg_u, deg_p in cases:
            errs = [
                _solve_one_step_poro_mms(
                    nx=nx,
                    backend=backend,
                    deg_v=deg_v,
                    deg_u=deg_u,
                    deg_p=deg_p,
                    kinv_case=kinv_case,
                )
                for nx in nx_list
            ]
            _print_convergence_table(
                backend=backend,
                degs=(deg_v, deg_u, deg_p),
                kinv_case=kinv_case,
                nx_list=list(nx_list),
                errs=errs,
            )
            for key in ("v", "u", "p"):
                series = [float(e[key]) for e in errs]
                assert all(np.isfinite(series))


def _select_fd_dofs(dh: DofHandler, fields_to_probe: dict[str, int], *, eid: int = 0) -> np.ndarray:
    selected: list[int] = []
    for field, count in fields_to_probe.items():
        try:
            local = dh.element_dofs(field, int(eid))
        except Exception:
            local = []
        selected.extend(list(local[:count]))
    return np.array(sorted(set(selected)), dtype=int)


def _fd_check(
    jac_form,
    res_form,
    dh: DofHandler,
    bcs,
    functions: dict[str, Function | VectorFunction],
    probe_dofs: np.ndarray,
    *,
    compiler: FormCompiler,
    eps: float = 1.0e-7,
):
    eq_jac = Equation(jac_form, None)
    eq_res = Equation(None, res_form)
    base_K, _ = compiler.assemble(eq_jac, bcs=bcs)
    if base_K is None:
        raise AssertionError("Jacobian assembly returned None.")

    direction = np.zeros(dh.total_dofs, dtype=float)
    field_dofs: dict[str, list[int]] = {}
    for gdof in probe_dofs:
        gdof_i = int(gdof)
        field, _ = dh._dof_to_node_map[gdof_i]
        if field not in functions:
            continue
        direction[gdof_i] = 1.0
        field_dofs.setdefault(field, []).append(gdof_i)

    if not np.any(direction):
        raise AssertionError("Directional FD probe is empty.")

    base_vals: dict[str, np.ndarray] = {}
    for field, dofs in field_dofs.items():
        dof_arr = np.asarray(dofs, dtype=int)
        base_vals[field] = functions[field].get_nodal_values(dof_arr)

    def _set_vals(sign: float):
        for field, dofs in field_dofs.items():
            dof_arr = np.asarray(dofs, dtype=int)
            functions[field].set_nodal_values(dof_arr, base_vals[field] + sign * eps)

    _set_vals(+1.0)
    _, R_plus = compiler.assemble(eq_res, bcs=bcs)
    _set_vals(-1.0)
    _, R_minus = compiler.assemble(eq_res, bcs=bcs)
    _set_vals(0.0)

    fd_vec = (R_plus - R_minus) / (2 * eps)
    jac_vec = base_K.dot(direction)
    err_vec = fd_vec - jac_vec
    max_abs = float(np.linalg.norm(err_vec, ord=np.inf))
    mag = float(np.linalg.norm(jac_vec, ord=np.inf))
    max_rel = max_abs / (mag + 1.0e-14) if mag > 0.0 else 0.0
    return max_abs, max_rel


@pytest.mark.parametrize("backend", _selected_backends(default="python"))
@pytest.mark.parametrize("kinv_case", ["iso", "aniso"])
def test_poro_eulerian_fd_residual_matches_jacobian_all_backends(backend, kinv_case):
    # Small mesh (full domain, no CutFEM here)
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=1, offset=(0.0, 0.0))
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )

    me = MixedElement(mesh, field_specs={"v_x": 1, "v_y": 1, "u_x": 1, "u_y": 1, "p": 1})
    dh = DofHandler(me, method="cg")

    Vv = FunctionSpace(name="v", field_names=["v_x", "v_y"], dim=1)
    Vu = FunctionSpace(name="u", field_names=["u_x", "u_y"], dim=1)

    dv = VectorTrialFunction(space=Vv, dof_handler=dh)
    du = VectorTrialFunction(space=Vu, dof_handler=dh)
    dp = TrialFunction(name="dp", field_name="p", dof_handler=dh)

    w = VectorTestFunction(space=Vv, dof_handler=dh)
    eta = VectorTestFunction(space=Vu, dof_handler=dh)
    q = TestFunction(name="q", field_name="p", dof_handler=dh)

    v_k = VectorFunction(name="v_k", field_names=["v_x", "v_y"], dof_handler=dh)
    u_k = VectorFunction(name="u_k", field_names=["u_x", "u_y"], dof_handler=dh)
    p_k = Function(name="p_k", field_name="p", dof_handler=dh)
    v_n = VectorFunction(name="v_n", field_names=["v_x", "v_y"], dof_handler=dh)
    u_n = VectorFunction(name="u_n", field_names=["u_x", "u_y"], dof_handler=dh)
    u_nm1 = VectorFunction(name="u_nm1", field_names=["u_x", "u_y"], dof_handler=dh)
    p_n = Function(name="p_n", field_name="p", dof_handler=dh)

    # Nontrivial state (small gradients so F=(I-∇u)^{-1} is well-defined)
    v_k.set_values_from_function(lambda x, y: np.array([0.1 * x + 0.05 * y, -0.03 * x + 0.02 * y]))
    u_k.set_values_from_function(lambda x, y: np.array([0.02 * x, -0.01 * y]))
    p_k.set_values_from_function(lambda x, y: 0.2 * x - 0.1 * y)
    v_n.set_values_from_function(lambda x, y: np.array([0.03 * x - 0.01 * y, 0.02 * x + 0.01 * y]))
    u_n.set_values_from_function(lambda x, y: np.array([0.01 * x, -0.005 * y]))
    u_nm1.nodal_values.fill(0.0)
    p_n.set_values_from_function(lambda x, y: 0.05 * x + 0.02 * y)

    qdeg = 6
    dx_p = dx(metadata={"q": qdeg})
    dt = Constant(0.1)
    theta = Constant(0.5)

    if str(kinv_case).strip().lower() in {"iso", "identity", "i"}:
        K_inv = Identity(2)
    else:
        K_inv = Constant(_kinv_ref_matrix(kinv_case).tolist(), dim=2)
        K_inv._jit_name = "K_inv"

    r = residual_poro(
        v_k,
        u_k,
        p_k,
        v_n,
        u_n,
        p_n,
        q,
        w,
        eta,
        u_nm1=u_nm1,
        rho_f=Constant(0.9),
        mu_f=Constant(1.1),
        rho_s0_tilde=Constant(1.0),
        phi=Constant(0.6),
        K_inv=K_inv,
        c_nh=Constant(0.7),
        beta_nh=Constant(0.0),
        dt=dt,
        theta=theta,
        dx_p=dx_p,
    )
    a = jacobian_poro(
        v_k,
        u_k,
        p_k,
        u_n,
        dv,
        du,
        dp,
        q,
        w,
        eta,
        u_nm1=u_nm1,
        rho_f=Constant(0.9),
        mu_f=Constant(1.1),
        rho_s0_tilde=Constant(1.0),
        phi=Constant(0.6),
        K_inv=K_inv,
        c_nh=Constant(0.7),
        beta_nh=Constant(0.0),
        dt=dt,
        theta=theta,
        dx_p=dx_p,
    )

    probe = _select_fd_dofs(dh, {"v_x": 1, "u_x": 1, "p": 1}, eid=0)
    compiler = FormCompiler(dh, quadrature_order=qdeg, backend=backend)

    fd_fields = {"v_x": v_k, "v_y": v_k, "u_x": u_k, "u_y": u_k, "p": p_k}
    abs_err, rel_err = _fd_check(
        a,
        r,
        dh,
        [],
        fd_fields,
        probe,
        compiler=compiler,
        eps=1.0e-7,
    )

    assert math.isfinite(abs_err) and math.isfinite(rel_err)
    assert abs_err < 1.0e-6
    assert rel_err < 1.0e-5
