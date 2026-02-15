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
        backend="python",
    )
    return float(np.asarray(res["mass"]).ravel()[0])


def _one_minus(expr):
    return (-expr) + Constant(1.0)


def _alpha_disk(x, y, *, cx: float = 0.5, cy: float = 0.5, R: float = 0.22, w: float = 0.03):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    # Smooth indicator: ~1 inside, ~0 outside.
    return 0.5 * (1.0 - np.tanh((r - R) / w))


def test_cahn_hilliard_preserves_alpha_mass():
    dt_val = 0.05
    n_steps = 4
    final_time = float(n_steps) * float(dt_val)

    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=6, ny=6, poly_order=1)
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
    xi = TestFunction("alpha", dof_handler=dh)
    eta = TestFunction("mu_alpha", dof_handler=dh)

    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    mu_k = Function("mu_k", "mu_alpha", dof_handler=dh)
    mu_n = Function("mu_n", "mu_alpha", dof_handler=dh)

    dt = Constant(float(dt_val))
    inv_dt = Constant(1.0 / float(dt_val))

    # Regularization parameters (robust on coarse meshes).
    M0 = Constant(0.2)
    gamma = Constant(1.0)
    eps = Constant(0.05)

    # Double-well: W(a)=a^2(1-a)^2 -> W'(a)=2a(1-a)(1-2a), W''(a)=2-12a+12a^2
    Wp = Constant(2.0) * alpha_k * _one_minus(alpha_k) * _one_minus(Constant(2.0) * alpha_k)
    Wpp = (-Constant(12.0) * alpha_k) + (Constant(12.0) * (alpha_k * alpha_k)) + Constant(2.0)

    # Backward Euler (theta=1) Cahn–Hilliard split system.
    r_alpha = xi * ((alpha_k - alpha_n) * inv_dt) * dx(metadata={"q": 4})
    r_alpha += inner(M0 * grad(mu_k), grad(xi)) * dx(metadata={"q": 4})

    r_mu = eta * mu_k * dx(metadata={"q": 4})
    r_mu += -(gamma * eps) * inner(grad(alpha_k), grad(eta)) * dx(metadata={"q": 4})
    r_mu += -eta * ((gamma / eps) * Wp) * dx(metadata={"q": 4})

    a_alpha = xi * (dalpha * inv_dt) * dx(metadata={"q": 4})
    a_alpha += inner(M0 * grad(dmu), grad(xi)) * dx(metadata={"q": 4})

    a_mu = eta * dmu * dx(metadata={"q": 4})
    a_mu += -(gamma * eps) * inner(grad(dalpha), grad(eta)) * dx(metadata={"q": 4})
    a_mu += -eta * ((gamma / eps) * Wpp * dalpha) * dx(metadata={"q": 4})

    residual_form = r_alpha + r_mu
    jacobian_form = a_alpha + a_mu

    solver = NewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(newton_tol=1.0e-8, max_newton_iter=25),
        quad_order=4,
        backend="python",
    )

    alpha_n.set_values_from_function(lambda x, y: _alpha_disk(x, y))
    mu_n.set_values_from_function(lambda x, y: 0.0)
    m0 = _mass(dh, alpha_n)

    solver.solve_time_interval(
        functions=[alpha_k, mu_k],
        prev_functions=[alpha_n, mu_n],
        aux_functions={"dt": dt},
        time_params=TimeStepperParameters(dt=dt_val, final_time=final_time, max_steps=n_steps, theta=1.0),
    )
    m1 = _mass(dh, alpha_k)

    rel_change = abs(m1 - m0) / max(1.0e-16, abs(m0))
    assert rel_change < 5.0e-6, f"Expected Cahn–Hilliard to preserve mass; got rel_change={rel_change:.2e}"

