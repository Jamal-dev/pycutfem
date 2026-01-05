import math
import numpy as np
import scipy.sparse.linalg as spla
import sympy as sp

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.analytic import Analytic, x, y
from pycutfem.ufl.expressions import (
    TrialFunction,
    TestFunction,
    VectorTrialFunction,
    VectorTestFunction,
    Constant,
    CellDiameter,
    FacetNormal,
    Jump,
    Neg,
    Pos,
    grad,
    inner,
    dot,
    div,
)
from pycutfem.ufl.forms import BoundaryCondition, assemble_form
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dx, ds, dS
from pycutfem.utils.meshgen import structured_quad


def solve_oseen(
    *,
    stabilization="none",
    nx=4,
    ny=4,
    nu=1.0e-2,
    u0_vec=(1.0, 0.0),
    quad_order=5,
    backend="jit",
    velocity_method="cg",
    pressure_method="cg",
    diffusion_flux="symmetric",
    pressure_jump=0.1,
):
    if stabilization not in {"none", "grad_div", "pressure_stab"}:
        raise ValueError(f"Unknown stabilization '{stabilization}'.")
    if velocity_method not in {"cg", "dg"}:
        raise ValueError(f"Unknown velocity_method '{velocity_method}'.")
    if pressure_method not in {"cg", "dg"}:
        raise ValueError(f"Unknown pressure_method '{pressure_method}'.")
    if diffusion_flux not in {"symmetric", "nonsymmetric"}:
        raise ValueError(f"Unknown diffusion_flux '{diffusion_flux}'.")

    u0_vec = np.asarray(u0_vec, dtype=float)
    u0 = Constant(u0_vec, dim=1)

    ux_sym = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    uy_sym = sp.cos(sp.pi * x) * sp.cos(sp.pi * y)
    p_sym = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)

    f_x = (
        -nu * sp.diff(ux_sym, x, 2)
        - nu * sp.diff(ux_sym, y, 2)
        + u0_vec[0] * sp.diff(ux_sym, x)
        + u0_vec[1] * sp.diff(ux_sym, y)
        + sp.diff(p_sym, x)
    )
    f_y = (
        -nu * sp.diff(uy_sym, x, 2)
        - nu * sp.diff(uy_sym, y, 2)
        + u0_vec[0] * sp.diff(uy_sym, x)
        + u0_vec[1] * sp.diff(uy_sym, y)
        + sp.diff(p_sym, y)
    )

    f_x_func = sp.lambdify((x, y), f_x, "numpy")
    f_y_func = sp.lambdify((x, y), f_y, "numpy")
    ux_exact = sp.lambdify((x, y), ux_sym, "numpy")
    uy_exact = sp.lambdify((x, y), uy_sym, "numpy")
    p_exact = sp.lambdify((x, y), p_sym, "numpy")
    stack_vec = lambda fx, fy: lambda xv, yv: np.stack([fx(xv, yv), fy(xv, yv)], axis=-1)
    f_vec = Analytic(stack_vec(f_x_func, f_y_func), degree=4, dim=1)
    u_exact_vec = Analytic(stack_vec(ux_exact, uy_exact), degree=4, dim=1)

    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )

    me = MixedElement(mesh, field_specs={"ux": 2, "uy": 2, "p": 1})
    dh = DofHandler(
        me,
        method="cg",
        field_methods={"ux": velocity_method, "uy": velocity_method, "p": pressure_method},
    )

    vel_space = FunctionSpace("velocity", ["ux", "uy"], dim=1)
    u = VectorTrialFunction(vel_space)
    v = VectorTestFunction(vel_space)
    p = TrialFunction("p")
    q = TestFunction("p")

    adv_strong = dot(dot(grad(u), u0), v)
    adv_weak = -dot(dot(grad(v), u0), u)
    if velocity_method == "cg":
        a = (nu * inner(grad(u), grad(v)) + adv_strong - p * div(v) + q * div(u)) * dx()
    else:
        a = (nu * inner(grad(u), grad(v)) + adv_weak - p * div(v) + q * div(u)) * dx()
    L = dot(f_vec, v) * dx()
    h = CellDiameter()
    if pressure_method == "dg":
        gauge = Constant(1.0e-6)
        a = a + gauge * p * q * dx()
        if pressure_jump > 0.0:
            tau_jump = Constant(pressure_jump)
            a = a + tau_jump / h * Jump(p) * Jump(q) * ds()

    if stabilization == "grad_div":
        gamma = Constant(1.0)
        a = a + gamma * div(u) * div(v) * dx()
    elif stabilization == "pressure_stab":
        tau_p = Constant(1.0e-3)
        a = a + tau_p * inner(grad(p), grad(q)) * dx()

    mesh.tag_boundary_edges({"boundary": lambda x, y: True})
    dh.tag_dof_by_locator("pressure_pin", "p", lambda x, y: np.isclose(x, 0.0) and np.isclose(y, 0.0))
    bcs = [
        BoundaryCondition("ux", "dirichlet", "boundary", ux_exact),
        BoundaryCondition("uy", "dirichlet", "boundary", uy_exact),
    ]
    if velocity_method == "dg":
        bcs = []
    if pressure_method == "cg":
        bcs.append(BoundaryCondition("p", "dirichlet", "pressure_pin", lambda x, y: p_exact(x, y)))

    if velocity_method == "dg":
        boundary_bs = mesh.edge_bitset("boundary")
        dS_bnd = dS(defined_on=boundary_bs)
        n = FacetNormal()
        vel_order = int(me._field_orders["ux"])
        penalty = Constant(10.0 * (vel_order + 1) ** 2)
        sym = 1.0 if diffusion_flux == "symmetric" else -1.0

        avg_grad_u = 0.5 * (Pos(grad(u)) + Neg(grad(u)))
        avg_grad_v = 0.5 * (Pos(grad(v)) + Neg(grad(v)))
        jump_u = Jump(u)
        jump_v = Jump(v)
        avg_u = 0.5 * (Pos(u) + Neg(u))

        beta_n = dot(u0, n)
        abs_beta_n = (beta_n * beta_n) ** Constant(0.5)

        a_int = (
            -nu * dot(dot(avg_grad_u, n), jump_v)
            -nu * sym * dot(dot(avg_grad_v, n), jump_u)
            + penalty * nu / h * dot(jump_u, jump_v)
            + beta_n * dot(avg_u, jump_v)
            + 0.5 * abs_beta_n * dot(jump_u, jump_v)
        ) * ds()

        a_bnd = (
            -nu * dot(dot(grad(u), n), v)
            -nu * sym * dot(dot(grad(v), n), u)
            + penalty * nu / h * dot(u, v)
            + 0.5 * (beta_n + abs_beta_n) * dot(u, v)
        ) * dS_bnd

        bnd_rhs = (
            -nu * sym * dot(grad(v), n)
            + penalty * nu / h * v
            + 0.5 * (beta_n - abs_beta_n) * v
        )
        L_bnd = dot(u_exact_vec, bnd_rhs) * dS_bnd

        a = a + a_int + a_bnd
        L = L + L_bnd

    K, F = assemble_form(
        a == L,
        dof_handler=dh,
        bcs=bcs,
        quad_order=quad_order,
        backend=backend,
    )
    sol = spla.spsolve(K, F)

    ux_dofs = np.asarray(dh.get_field_slice("ux"), dtype=int)
    uy_dofs = np.asarray(dh.get_field_slice("uy"), dtype=int)
    p_dofs = np.asarray(dh.get_field_slice("p"), dtype=int)

    ux_coords = dh.get_dof_coords("ux")
    uy_coords = dh.get_dof_coords("uy")
    p_coords = dh.get_dof_coords("p")

    ux_err = np.sqrt(np.mean((sol[ux_dofs] - ux_exact(ux_coords[:, 0], ux_coords[:, 1])) ** 2))
    uy_err = np.sqrt(np.mean((sol[uy_dofs] - uy_exact(uy_coords[:, 0], uy_coords[:, 1])) ** 2))
    p_vals = sol[p_dofs]
    p_ref = p_exact(p_coords[:, 0], p_coords[:, 1])
    if pressure_method == "dg":
        p_vals = p_vals - np.mean(p_vals)
        p_ref = p_ref - np.mean(p_ref)
    p_err = np.sqrt(np.mean((p_vals - p_ref) ** 2))

    u_err = float(np.sqrt(ux_err ** 2 + uy_err ** 2))
    return u_err, float(p_err)


def convergence_study(
    *,
    base_n=2,
    levels=4,
    stabilization="grad_div",
    nu=1.0e-2,
    quad_order=4,
    backend="python",
    velocity_method="cg",
    pressure_method="cg",
    diffusion_flux="symmetric",
    pressure_jump=0.1,
):
    ns = [int(base_n * (2 ** i)) for i in range(levels)]
    u_errs = []
    p_errs = []
    for n in ns:
        u_err, p_err = solve_oseen(
            stabilization=stabilization,
            nx=n,
            ny=n,
            nu=nu,
            quad_order=quad_order,
            backend=backend,
            velocity_method=velocity_method,
            pressure_method=pressure_method,
            diffusion_flux=diffusion_flux,
            pressure_jump=pressure_jump,
        )
        u_errs.append(u_err)
        p_errs.append(p_err)

    u_ooa = []
    p_ooa = []
    for i in range(1, len(ns)):
        if u_errs[i] > 0.0 and u_errs[i - 1] > 0.0:
            u_ooa.append(math.log(u_errs[i - 1] / u_errs[i], 2.0))
        else:
            u_ooa.append(float("nan"))
        if p_errs[i] > 0.0 and p_errs[i - 1] > 0.0:
            p_ooa.append(math.log(p_errs[i - 1] / p_errs[i], 2.0))
        else:
            p_ooa.append(float("nan"))

    return {"ns": ns, "u_errs": u_errs, "p_errs": p_errs, "u_ooa": u_ooa, "p_ooa": p_ooa}


def main():
    for stab in ("none", "grad_div", "pressure_stab"):
        u_err, p_err = solve_oseen(stabilization=stab, nx=4, ny=4, nu=1.0e-2)
        print(f"{stab:>13s} | velocity RMS error = {u_err:.3e}, pressure RMS error = {p_err:.3e}")


if __name__ == "__main__":
    main()
