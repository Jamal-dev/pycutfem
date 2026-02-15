import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
from pycutfem.ufl.expressions import Constant, Function, Laplacian, TestFunction, TrialFunction, grad, inner
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


def _build_allen_cahn_problem(*, nx: int, dt_val: float, conservative: bool):
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=nx, ny=nx, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )

    field_specs = {"alpha": 1}
    if conservative:
        field_specs["lambda_alpha"] = ":number:"

    me = MixedElement(mesh, field_specs=field_specs)
    dh = DofHandler(me, method="cg")

    dalpha = TrialFunction("alpha", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)

    dt = Constant(float(dt_val))
    inv_dt = Constant(1.0 / float(dt_val))

    # Regularization parameters (chosen for robustness on coarse meshes).
    M = Constant(1.0)
    gamma = Constant(1.0)
    eps = Constant(0.05)

    def _one_minus(expr):
        return (-expr) + Constant(1.0)

    # Double-well: W(a)=a^2(1-a)^2 -> W'(a)=2a(1-a)(1-2a), W''(a)=2-12a+12a^2
    Wp = Constant(2.0) * alpha_k * _one_minus(alpha_k) * _one_minus(Constant(2.0) * alpha_k)
    Wpp = (-Constant(12.0) * alpha_k) + (Constant(12.0) * (alpha_k * alpha_k)) + Constant(2.0)

    r_alpha = alpha_test * ((alpha_k - alpha_n) * inv_dt) * dx(metadata={"q": 4})
    r_alpha += (M * gamma * eps) * inner(grad(alpha_k), grad(alpha_test)) * dx(metadata={"q": 4})
    r_alpha += alpha_test * ((M * gamma / eps) * Wp) * dx(metadata={"q": 4})

    a_alpha = alpha_test * (dalpha * inv_dt) * dx(metadata={"q": 4})
    a_alpha += (M * gamma * eps) * inner(grad(dalpha), grad(alpha_test)) * dx(metadata={"q": 4})
    a_alpha += alpha_test * ((M * gamma / eps) * Wpp * dalpha) * dx(metadata={"q": 4})

    residual_form = r_alpha
    jacobian_form = a_alpha

    lambda_alpha_k = None
    lambda_alpha_n = None
    if conservative:
        dlambda = TrialFunction("lambda_alpha", dof_handler=dh)
        lambda_test = TestFunction("lambda_alpha", dof_handler=dh)
        lambda_alpha_k = Function("lambda_alpha_k", "lambda_alpha", dof_handler=dh)
        lambda_alpha_n = Function("lambda_alpha_n", "lambda_alpha", dof_handler=dh)

        # Conservative correction in alpha equation.
        residual_form += -alpha_test * (M * lambda_alpha_k) * dx(metadata={"q": 4})
        jacobian_form += -alpha_test * (M * dlambda) * dx(metadata={"q": 4})

        # Global constraint: ∫ M (μ_α - λ_α) dx = 0 with μ_α = γ(-εΔα + (1/ε)W'(α)).
        mu_alpha = gamma * ((-eps) * Laplacian(alpha_k) + (Wp / eps))
        dmu_alpha = gamma * ((-eps) * Laplacian(dalpha) + (Wpp * dalpha / eps))
        residual_form += lambda_test * (M * (mu_alpha - lambda_alpha_k)) * dx(metadata={"q": 4})
        jacobian_form += lambda_test * (M * (dmu_alpha - dlambda)) * dx(metadata={"q": 4})

    solver = NewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(newton_tol=1.0e-8, max_newton_iter=20),
        quad_order=4,
        backend="python",
    )

    return dh, solver, dt, alpha_k, alpha_n, lambda_alpha_k, lambda_alpha_n


def _alpha_disk(x, y, *, cx: float = 0.5, cy: float = 0.5, R: float = 0.22, w: float = 0.03):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    # Smooth indicator: ~1 inside, ~0 outside.
    return 0.5 * (1.0 - np.tanh((r - R) / w))


def test_conservative_allen_cahn_preserves_alpha_mass():
    dt_val = 0.05
    n_steps = 4
    final_time = float(n_steps) * float(dt_val)

    # Non-conservative Allen–Cahn: mass should decrease (curvature-driven shrinkage).
    dh_nc, solver_nc, dt_nc, alpha_k_nc, alpha_n_nc, _, _ = _build_allen_cahn_problem(nx=6, dt_val=dt_val, conservative=False)
    alpha_n_nc.set_values_from_function(lambda x, y: _alpha_disk(x, y))
    m0_nc = _mass(dh_nc, alpha_n_nc)

    solver_nc.solve_time_interval(
        functions=[alpha_k_nc],
        prev_functions=[alpha_n_nc],
        aux_functions={"dt": dt_nc},
        time_params=TimeStepperParameters(dt=dt_val, final_time=final_time, max_steps=n_steps, theta=1.0),
    )
    m1_nc = _mass(dh_nc, alpha_k_nc)

    # Conservative Allen–Cahn: mass should remain essentially constant.
    dh_c, solver_c, dt_c, alpha_k_c, alpha_n_c, lambda_k, lambda_n = _build_allen_cahn_problem(nx=6, dt_val=dt_val, conservative=True)
    alpha_n_c.set_values_from_function(lambda x, y: _alpha_disk(x, y))
    lambda_n.nodal_values.fill(0.0)
    m0_c = _mass(dh_c, alpha_n_c)

    solver_c.solve_time_interval(
        functions=[alpha_k_c, lambda_k],
        prev_functions=[alpha_n_c, lambda_n],
        aux_functions={"dt": dt_c},
        time_params=TimeStepperParameters(dt=dt_val, final_time=final_time, max_steps=n_steps, theta=1.0),
    )
    m1_c = _mass(dh_c, alpha_k_c)

    rel_drop_nc = (m0_nc - m1_nc) / max(1.0e-16, abs(m0_nc))
    rel_change_c = abs(m1_c - m0_c) / max(1.0e-16, abs(m0_c))

    assert rel_drop_nc > 5.0e-4, f"Expected non-conservative Allen–Cahn mass to drop; got rel_drop={rel_drop_nc:.2e}"
    assert rel_change_c < 5.0e-4, f"Expected conservative Allen–Cahn mass to be preserved; got rel_change={rel_change_c:.2e}"
