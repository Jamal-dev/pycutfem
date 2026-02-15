import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction, grad, inner
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad


def _mass(dh: DofHandler, alpha: Function, *, q: int = 4) -> float:
    res = assemble_form(
        Equation(alpha * dx(metadata={"q": int(q)}), None),
        dof_handler=dh,
        assembler_hooks={alpha: {"name": "mass"}},
        backend="cpp",
    )
    return float(np.asarray(res["mass"]).ravel()[0])


def _alpha_disk(x, y, *, cx: float = 0.5, cy: float = 0.5, R: float = 0.22, w: float = 0.03):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    # Smooth indicator: ~1 inside, ~0 outside.
    return 0.5 * (1.0 - np.tanh((r - R) / w))


def _build_diffusion_problem(*, nx: int, dt_val: float, D: float, backend: str = "cpp"):
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=nx, ny=nx, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )

    me = MixedElement(mesh, field_specs={"alpha": 1})
    dh = DofHandler(me, method="cg")

    dalpha = TrialFunction("alpha", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)

    dt = Constant(float(dt_val))
    inv_dt = Constant(1.0 / float(dt_val))
    D_c = Constant(float(D))

    r_alpha = alpha_test * ((alpha_k - alpha_n) * inv_dt) * dx(metadata={"q": 4})
    r_alpha += D_c * inner(grad(alpha_k), grad(alpha_test)) * dx(metadata={"q": 4})

    a_alpha = alpha_test * (dalpha * inv_dt) * dx(metadata={"q": 4})
    a_alpha += D_c * inner(grad(dalpha), grad(alpha_test)) * dx(metadata={"q": 4})

    solver = NewtonSolver(
        r_alpha,
        a_alpha,
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(newton_tol=1.0e-10, max_newton_iter=10),
        quad_order=4,
        backend=str(backend),
    )
    return dh, solver, dt, alpha_k, alpha_n


def _build_cahn_hilliard_problem(
    *,
    nx: int,
    dt_val: float,
    M0: float,
    gamma: float,
    eps: float,
    mobility: str,
    backend: str = "cpp",
):
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=nx, ny=nx, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )

    me = MixedElement(mesh, field_specs={"alpha": 1, "mu_alpha": 1})
    dh = DofHandler(me, method="cg")

    dalpha = TrialFunction("alpha", dof_handler=dh)
    dmu = TrialFunction("mu_alpha", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    mu_test = TestFunction("mu_alpha", dof_handler=dh)

    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    mu_k = Function("mu_k", "mu_alpha", dof_handler=dh)
    mu_n = Function("mu_n", "mu_alpha", dof_handler=dh)

    dt = Constant(float(dt_val))
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
    elif mob_key in {"degenerate", "deg", "alpha(1-alpha)", "alpha*(1-alpha)", "a(1-a)"}:
        mob = alpha_k * _one_minus(alpha_k)
        mob_prime = _one_minus(Constant(2.0) * alpha_k)  # 1-2a
        M_k = M0_c * mob
        dM = M0_c * mob_prime * dalpha
    else:
        raise ValueError(f"Unknown mobility={mobility!r}. Use 'constant' or 'degenerate'.")

    # Double-well derivatives for W(a)=a^2(1-a)^2.
    Wp = Constant(2.0) * alpha_k * _one_minus(alpha_k) * _one_minus(Constant(2.0) * alpha_k)
    Wpp = (-Constant(12.0) * alpha_k) + (Constant(12.0) * (alpha_k * alpha_k)) + Constant(2.0)

    # Backward Euler (theta=1): alpha_t = div(M grad(mu)).
    r_alpha = alpha_test * ((alpha_k - alpha_n) * inv_dt) * dx(metadata={"q": 4})
    r_alpha += inner(M_k * grad(mu_k), grad(alpha_test)) * dx(metadata={"q": 4})

    # mu = gamma(-eps Δalpha + (1/eps) W'(alpha))
    r_mu = mu_test * mu_k * dx(metadata={"q": 4})
    r_mu += -(gamma_c * eps_c) * inner(grad(alpha_k), grad(mu_test)) * dx(metadata={"q": 4})
    r_mu += -mu_test * ((gamma_c / eps_c) * Wp) * dx(metadata={"q": 4})

    residual_form = r_alpha + r_mu

    a_alpha = alpha_test * (dalpha * inv_dt) * dx(metadata={"q": 4})
    a_alpha += (dM * inner(grad(mu_k), grad(alpha_test))) * dx(metadata={"q": 4})
    a_alpha += inner(M_k * grad(dmu), grad(alpha_test)) * dx(metadata={"q": 4})

    a_mu = mu_test * dmu * dx(metadata={"q": 4})
    a_mu += -(gamma_c * eps_c) * inner(grad(dalpha), grad(mu_test)) * dx(metadata={"q": 4})
    a_mu += -mu_test * ((gamma_c / eps_c) * Wpp * dalpha) * dx(metadata={"q": 4})

    jacobian_form = a_alpha + a_mu

    solver = NewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(newton_tol=1.0e-9, max_newton_iter=25),
        quad_order=4,
        backend=str(backend),
    )

    return dh, solver, dt, alpha_k, alpha_n, mu_k, mu_n


def test_cahn_hilliard_preserves_mass_and_prevents_diffusive_receding():
    dt_val = 0.1
    n_steps = 2
    final_time = float(n_steps) * float(dt_val)
    nx = 6

    # Diffusion-only: should smear the indicator (max(alpha) drops noticeably).
    dh_diff, solver_diff, dt_diff, alpha_k_diff, alpha_n_diff = _build_diffusion_problem(nx=nx, dt_val=dt_val, D=0.5)
    alpha_n_diff.set_values_from_function(lambda x, y: _alpha_disk(x, y))
    m0_diff = _mass(dh_diff, alpha_n_diff)
    amax0 = float(np.max(alpha_n_diff.nodal_values))

    solver_diff.solve_time_interval(
        functions=[alpha_k_diff],
        prev_functions=[alpha_n_diff],
        aux_functions={"dt": dt_diff},
        time_params=TimeStepperParameters(dt=dt_val, final_time=final_time, max_steps=n_steps, theta=1.0),
    )
    m1_diff = _mass(dh_diff, alpha_k_diff)
    amax_diff = float(np.max(alpha_k_diff.nodal_values))

    # Cahn–Hilliard (degenerate mobility): mass-conserving and interface-localized relaxation.
    dh_ch, solver_ch, dt_ch, alpha_k_ch, alpha_n_ch, mu_k, mu_n = _build_cahn_hilliard_problem(
        nx=nx,
        dt_val=dt_val,
        M0=0.4,
        gamma=1.0,
        eps=0.05,
        mobility="degenerate",
    )
    alpha_n_ch.set_values_from_function(lambda x, y: _alpha_disk(x, y))
    # Good initial guess for mu: ignore the Laplacian term and use only (gamma/eps) W'(alpha).
    a0 = alpha_n_ch.nodal_values
    mu_n.nodal_values[:] = (1.0 / 0.05) * (2.0 * a0 * (1.0 - a0) * (1.0 - 2.0 * a0))
    m0_ch = _mass(dh_ch, alpha_n_ch)

    solver_ch.solve_time_interval(
        functions=[alpha_k_ch, mu_k],
        prev_functions=[alpha_n_ch, mu_n],
        aux_functions={"dt": dt_ch},
        time_params=TimeStepperParameters(dt=dt_val, final_time=final_time, max_steps=n_steps, theta=1.0),
    )
    m1_ch = _mass(dh_ch, alpha_k_ch)
    amax_ch = float(np.max(alpha_k_ch.nodal_values))

    # Mass should remain essentially constant for Cahn–Hilliard (no sources/sinks).
    rel_mass_change_ch = abs(m1_ch - m0_ch) / max(1.0e-16, abs(m0_ch))
    assert rel_mass_change_ch < 5.0e-4, f"Expected CH mass conservation; got rel_change={rel_mass_change_ch:.2e}"

    # "No receding": CH keeps the saturated phase near 1, while diffusion reduces the peak substantially.
    assert amax_diff < 0.95 * amax0
    assert amax_ch > 0.97 * amax0

    # Diffusion still conserves mass under natural (no-flux) boundary conditions in this setup.
    rel_mass_change_diff = abs(m1_diff - m0_diff) / max(1.0e-16, abs(m0_diff))
    assert rel_mass_change_diff < 5.0e-4
