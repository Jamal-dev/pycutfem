#!/usr/bin/env python
# coding: utf-8
"""
Verification of the 2-D Neo-Hookean element with a manufactured solution.

- Material: compressible Neo-Hookean with the Cauchy/PK2 model from
  `examples/neo-hook-2d-mat-model.py` (C10=0.5, kappa=500).
- PDE: -div(P) = b on the unit square with Dirichlet data from the exact
  displacement d_exact = A * [sin(pi x) sin(pi y), sin(pi y) sin(pi x)],
  with A = 0.052. Dirichlet is applied on the whole boundary, so the boundary
  traction term drops.
- Discretisation: Qk/Qk displacement, solved with Newton's method.
- Jacobian: analytic linearisation of the PK1 tensor; a finite-difference
  probe is run to confirm alignment.
- Verification: mesh refinement study with L2 and H1 errors against the
  manufactured solution.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import sympy as sp

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import CircleLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import (
    NewtonParameters,
    NewtonSolver,
    TimeStepperParameters,
)
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    Identity,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    det,
    dot,
    grad,
    inner,
    inv,
    trace,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad

try:
    from pycutfem.io.vtk import export_vtk
except Exception:  # pragma: no cover - optional dependency
    export_vtk = None


# -----------------------------------------------------------------------------
# Manufactured data (exact displacement, body force, traction)
# -----------------------------------------------------------------------------


@dataclass
class ManufacturedData:
    exact_disp: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    exact_grad: Callable[[np.ndarray, np.ndarray], np.ndarray]
    body_force: Callable[[np.ndarray, np.ndarray], np.ndarray]
    traction: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def build_manufactured_data(amplitude: float, c10: float, kappa: float) -> ManufacturedData:
    """
    Build exact displacement, gradient, body force and traction using SymPy.
    """
    x, y = sp.symbols("x y")
    d1 = amplitude * sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    d2 = amplitude * sp.sin(sp.pi * y) * sp.sin(sp.pi * x)

    F = sp.Matrix(
        [
            [1 + sp.diff(d1, x), sp.diff(d1, y)],
            [sp.diff(d2, x), 1 + sp.diff(d2, y)],
        ]
    )
    C = F.T * F
    A = C.inv()
    I1 = sp.trace(C)
    J = sp.sqrt(C.det())

    S_iso = (2 * c10 / J) * (sp.eye(2) - sp.Rational(1, 2) * I1 * A)
    p = kappa * (J - 1)
    S_vol = J * p * A
    S = S_iso + S_vol
    P = F * S

    divP = sp.Matrix([sp.diff(P[0, 0], x) + sp.diff(P[0, 1], y), sp.diff(P[1, 0], x) + sp.diff(P[1, 1], y)])
    b = -divP
    grad_d = sp.Matrix(
        [
            [sp.diff(d1, x), sp.diff(d1, y)],
            [sp.diff(d2, x), sp.diff(d2, y)],
        ]
    )

    disp_fun = sp.lambdify((x, y), (d1, d2), "numpy")
    grad_fun = sp.lambdify((x, y), grad_d, "numpy")
    body_fun = sp.lambdify((x, y), (b[0], b[1]), "numpy")
    P_fun = sp.lambdify((x, y), P, "numpy")

    def _vector_callable(fn):
        def wrapper(xv, yv):
            v0, v1 = fn(xv, yv)
            return np.stack([v0, v1], axis=-1)

        return wrapper

    def traction_fn(xv, yv, normal: np.ndarray):
        px = np.asarray(P_fun(xv, yv), dtype=float)
        return px @ normal

    return ManufacturedData(
        exact_disp=_vector_callable(disp_fun),
        exact_grad=lambda xv, yv: np.asarray(grad_fun(xv, yv), dtype=float),
        body_force=_vector_callable(body_fun),
        traction=traction_fn,
    )


# -----------------------------------------------------------------------------
# Neo-Hookean material (PK1 and tangent)
# -----------------------------------------------------------------------------


def neo_hookean_pk1(F, *, c10: Constant, kappa: Constant):
    """
    First Piola-Kirchhoff tensor and cached invariants for the compressible
    Neo-Hookean model defined in examples/neo-hook-2d-mat-model.py.
    """
    I2 = Identity(2)
    C = dot(F.T, F)
    A = inv(C)
    I1 = trace(C)
    J = det(F)

    S_iso = (Constant(2.0) * c10 / J) * (I2 - Constant(0.5) * I1 * A)
    p = kappa * (J - Constant(1.0))
    S_vol = J * p * A
    S = S_iso + S_vol
    P = dot(F, S)
    cache = {"C": C, "A": A, "I1": I1, "J": J, "S": S}
    return P, cache


def neo_hookean_delta_P(F, cache: Dict[str, object], delta_F, *, c10: Constant, kappa: Constant):
    """
    Gateaux derivative dP[delta_F] for the Neo-Hookean PK1 tensor.
    """
    I2 = Identity(2)
    C = cache["C"]
    A = cache["A"]
    I1 = cache["I1"]
    J = cache["J"]
    S = cache["S"]

    delta_C = dot(delta_F.T, F) + dot(F.T, delta_F)
    tr_deltaC = trace(delta_C)
    tr_A_deltaC = trace(dot(A, delta_C))

    mu2 = Constant(2.0) * c10
    term_I = I2 - Constant(0.5) * I1 * A
    deltaS_iso = (
        -(mu2 / (Constant(2.0) * J)) * tr_A_deltaC * term_I
        + (mu2 / J) * (-Constant(0.5) * tr_deltaC * A + Constant(0.5) * I1 * dot(dot(A, delta_C), A))
    )

    alpha = kappa * J * (J - Constant(1.0))
    deltaS_vol = (
        Constant(0.5) * kappa * J * (Constant(2.0) * J - Constant(1.0)) * tr_A_deltaC * A
        - alpha * dot(dot(A, delta_C), A)
    )

    deltaS = deltaS_iso + deltaS_vol
    deltaP = dot(delta_F, S) + dot(F, deltaS)
    return deltaP


# -----------------------------------------------------------------------------
# Finite-difference Jacobian check
# -----------------------------------------------------------------------------


def pick_probe_dofs(dh: DofHandler, bcs: Iterable[BoundaryCondition], per_field: int = 2) -> np.ndarray:
    bc_dofs = set(dh.get_dirichlet_data(bcs).keys())
    probes: list[int] = []
    for field in dh.mixed_element.field_names:
        taken = 0
        for gd in dh.get_field_slice(field):
            if int(gd) in bc_dofs:
                continue
            probes.append(int(gd))
            taken += 1
            if taken >= per_field:
                break
    return np.array(probes, dtype=int)


def finite_difference_check(
    jac_form,
    res_form,
    dh: DofHandler,
    bcs: Iterable[BoundaryCondition],
    functions: Dict[str, VectorFunction | Function],
    probe_dofs: Iterable[int],
    eps: float = 1.0e-6,
    backend: str = "jit",
) -> None:
    compiler = FormCompiler(dh, backend=backend)
    eq = Equation(jac_form, res_form)
    base_K, base_R = compiler.assemble(eq, bcs=bcs)
    if base_K is None or base_R is None:
        print("Skipping FD check: Jacobian or residual form missing.")
        return

    def perturb(field: str, gdof: int, new_value: float) -> float:
        func = functions[field]
        old = func.get_nodal_values(np.array([gdof], dtype=int))[0]
        func.set_nodal_values(np.array([gdof], dtype=int), np.array([new_value], dtype=float))
        return old

    bc_dofs = set(dh.get_dirichlet_data(bcs).keys())
    rows = []
    for gdof in probe_dofs:
        field, _ = dh._dof_to_node_map[int(gdof)]
        if field not in functions:
            continue
        if int(gdof) in bc_dofs:
            continue
        old_val = functions[field].get_nodal_values(np.array([gdof], dtype=int))[0]
        perturb(field, int(gdof), old_val + eps)
        K_plus, R_plus = compiler.assemble(eq, bcs=bcs)
        perturb(field, int(gdof), old_val - eps)
        K_minus, R_minus = compiler.assemble(eq, bcs=bcs)
        perturb(field, int(gdof), old_val)
        fd_col = (R_plus - R_minus) / (2 * eps)
        jac_col = base_K[:, int(gdof)].toarray().ravel()
        err_vec = fd_col - jac_col
        err = np.linalg.norm(err_vec, ord=np.inf)
        mag = np.linalg.norm(jac_col, ord=np.inf)
        rel = err / (mag + 1.0e-14)
        rows.append((gdof, field, err, mag, rel))
    print("Finite-difference Jacobian check (gdof, field, err, |J|, rel):")
    for gd, fld, err, mag, rel in rows:
        print(f"  {gd:5d}  {fld:6s}  err={err:9.3e}  |J|={mag:9.3e}  rel={rel:9.3e}")


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------


def run_verification(*, degree: int = 2, cycles: int = 3, backend: str = "jit", run_fd: bool = True, export: bool = False):
    amplitude = 0.052
    c10_val = 0.5
    kappa_val = 50.0
    manufactured = build_manufactured_data(amplitude, c10_val, kappa_val)

    c10 = Constant(c10_val)
    kappa = Constant(kappa_val)
    load_steps = 5  # number of load increments
    dt = 1.0 / load_steps

    disp_space = FunctionSpace("disp", ["ux", "uy"])
    convergence = []

    for cycle in range(cycles):
        nx = ny = 8 * (2**cycle)
        nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=degree)
        mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=degree)

        bc_tags = {
            "left": lambda x, y: np.isclose(x, 0.0),
            "right": lambda x, y: np.isclose(x, 1.0),
            "bottom": lambda x, y: np.isclose(y, 0.0),
            "top": lambda x, y: np.isclose(y, 1.0),
        }
        mesh.tag_boundary_edges(bc_tags)

        me = MixedElement(mesh, field_specs={"ux": degree, "uy": degree})
        dh = DofHandler(me, method="cg")

        disp = VectorFunction(name="d", field_names=["ux", "uy"], dof_handler=dh)
        disp_prev = VectorFunction(name="d_prev", field_names=["ux", "uy"], dof_handler=dh)
        disp_prev.nodal_values.fill(0.0)

        # Dirichlet BCs (ramped exact solution) and their homogeneous counterparts
        load_factor = Constant(0.0)  # updated every load step
        exact_x = lambda x, y, t=0.0: float(load_factor.value * amplitude * math.sin(math.pi * x) * math.sin(math.pi * y))
        exact_y = lambda x, y, t=0.0: float(load_factor.value * amplitude * math.sin(math.pi * y) * math.sin(math.pi * x))
        bcs = [
            BoundaryCondition("ux", "dirichlet", tag, exact_x) for tag in bc_tags
        ] + [
            BoundaryCondition("uy", "dirichlet", tag, exact_y) for tag in bc_tags
        ]
        zero_bc = lambda x, y, t=0.0: 0.0
        bcs_homog = [
            BoundaryCondition("ux", "dirichlet", tag, zero_bc) for tag in bc_tags
        ] + [
            BoundaryCondition("uy", "dirichlet", tag, zero_bc) for tag in bc_tags
        ]

        du = VectorTrialFunction(space=disp_space, dof_handler=dh)
        w = VectorTestFunction(space=disp_space, dof_handler=dh)

        I2 = Identity(2)
        dx_vol = dx(metadata={"q": 2 * degree + 4})

        F = I2 + grad(disp)
        P, cache = neo_hookean_pk1(F, c10=c10, kappa=kappa)
        deltaP = neo_hookean_delta_P(F, cache, grad(du), c10=c10, kappa=kappa)

        body_force_expr = load_factor * Analytic(manufactured.body_force)
        residual_form = (inner(P, grad(w)) - dot(body_force_expr, w)) * dx_vol
        jacobian_form = inner(deltaP, grad(w)) * dx_vol

        if run_fd and cycle == 0:
            load_factor.value = dt  # probe the Jacobian at the first load increment
            probe = pick_probe_dofs(dh, bcs_homog, per_field=2)
            fd_fields = {"ux": disp, "uy": disp}
            finite_difference_check(jacobian_form, residual_form, dh, bcs_homog, fd_fields, probe, eps=1.0e-7, backend=backend)
            load_factor.value = 0.0  # start the ramp from zero load

        # Load ramp: solve sequentially for t = k*dt, k=1..load_steps (no internal adaptive dt)
        for k in range(1, load_steps + 1):
            load_factor.value = k * dt
            solver = NewtonSolver(
                residual_form=residual_form,
                jacobian_form=jacobian_form,
                dof_handler=dh,
                mixed_element=me,
                bcs=bcs,
                bcs_homog=bcs_homog,
                newton_params=NewtonParameters(newton_tol=1e-10, max_newton_iter=15, line_search=True),
                backend=backend,
            )
            time_params = TimeStepperParameters(dt=1.0, max_steps=1, stop_on_steady=False)
            solver.solve_time_interval(
                functions=[disp],
                prev_functions=[disp_prev],
                time_params=time_params,
            )

        level_set = CircleLevelSet(center=(0.5, 0.5), radius=10.0)
        exact_map = {"ux": lambda x, y: manufactured.exact_disp(x, y)[..., 0], "uy": lambda x, y: manufactured.exact_disp(x, y)[..., 1]}
        l2_error = dh.l2_error_on_side(functions=disp, exact=exact_map, level_set=level_set, side="-", quad_order=2 * degree + 6)
        h1_error = dh.h1_error_vector_on_side(
            Uh=disp,
            exact_grad_vec=lambda x, y: manufactured.exact_grad(x, y),
            level_set=level_set,
            side="-",
            quad_order=2 * degree + 6,
        )

        h = max(1.0 / nx, 1.0 / ny)
        convergence.append({"cycle": cycle, "h": h, "L2": l2_error, "H1": h1_error, "ndofs": dh.total_dofs})

        if export and export_vtk is not None and cycle == cycles - 1:
            os.makedirs("neo_hookean_results", exist_ok=True)
            path = os.path.join("neo_hookean_results", f"solution_cycle{cycle}.vtu")
            export_vtk(filename=path, mesh=mesh, dof_handler=dh, functions={"displacement": disp})

        print(f"[cycle {cycle}] h={h:.3e}  ndofs={dh.total_dofs}  L2={l2_error:.3e}  H1={h1_error:.3e}")

    print("\nConvergence (Neo-Hookean MMS)")
    print(f"{'cycle':>5} | {'h':>10} | {'ndofs':>8} | {'L2':>12} | {'rate':>6} | {'H1':>12} | {'rate':>6}")
    print("-" * 70)
    for i, row in enumerate(convergence):
        rate_l2 = "----"
        rate_h1 = "----"
        if i > 0:
            prev = convergence[i - 1]
            rate_l2 = f"{math.log(prev['L2'] / row['L2'], 2):.2f}"
            rate_h1 = f"{math.log(prev['H1'] / row['H1'], 2):.2f}"
        print(
            f"{row['cycle']:5d} | {row['h']:10.3e} | {row['ndofs']:8d} | {row['L2']:12.4e} | {rate_l2:>6} | "
            f"{row['H1']:12.4e} | {rate_h1:>6}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neo-Hookean manufactured-solution verification")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree for displacement (Qk).")
    parser.add_argument("--cycles", type=int, default=3, help="Number of refinement cycles.")
    parser.add_argument("--backend", choices=("jit", "python"), default="jit", help="Assembly backend.")
    parser.add_argument("--no-fd", dest="run_fd", action="store_false", help="Skip finite-difference Jacobian check on the coarsest mesh.")
    parser.add_argument("--export", action="store_true", help="Write VTU output for the last cycle.")
    args, _ = parser.parse_known_args()
    run_verification(degree=args.degree, cycles=args.cycles, backend=args.backend, run_fd=args.run_fd, export=args.export)
