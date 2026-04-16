#!/usr/bin/env python3
"""Diffuse Stokes/Darcy straight-interface debug benchmark based on Stoter et al.

This driver implements the 2D manufactured straight-interface problem described
in `stoter_diffusive_interface_bjs.tex`, Section 5.2. It is intentionally
benchmark-local and compact:

  - Stokes unknowns `(u, p)` on the full embedding domain,
  - Darcy potential `phi_D` on the full embedding domain,
  - diffuse interface terms assembled exactly in the Stoter style,
  - sharp outer Dirichlet data from the manufactured exact solution,
  - structured meshes for fast screening and convergence checks.

The goal is diagnostic, not paper-ready production. If this driver does not
recover the manufactured solution with refinement, then the diffuse interface
coupling logic is still wrong independently of the Seboldt Stokes--Biot model.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import scipy.sparse.linalg as spla

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction, div, dot, grad, inner
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def _eps(v):
    return Constant(0.5) * (grad(v) + grad(v).T)


def _tag_boundaries(mesh: Mesh, *, Lx: float, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "stokes_top": lambda x, y: abs(y - 1.0) <= tol,
            "stokes_left": lambda x, y: abs(x - 0.0) <= tol and y >= -tol,
            "stokes_right": lambda x, y: abs(x - Lx) <= tol and y >= -tol,
            "darcy_bottom": lambda x, y: abs(y + 1.0) <= tol,
            "darcy_left": lambda x, y: abs(x - 0.0) <= tol and y <= tol,
            "darcy_right": lambda x, y: abs(x - Lx) <= tol and y <= tol,
        }
    )


def _stack_vec(fx, fy):
    return lambda x, y: np.stack([fx(x, y), fy(x, y)], axis=-1)


def _stoter_exact_fields(*, nu: float, K: float, friction_alpha: float, g: float, p_const: float):
    coeff_quad = 0.5 * float(K) - (float(friction_alpha) * float(g)) / (4.0 * float(nu) * float(nu))

    def u_profile(y):
        yv = np.asarray(y, dtype=float)
        return -float(K) - (float(g) * yv) / (2.0 * float(nu)) + coeff_quad * (yv * yv)

    def du_dy(y):
        yv = np.asarray(y, dtype=float)
        return -(float(g)) / (2.0 * float(nu)) + 2.0 * coeff_quad * yv

    ux = lambda x, y: du_dy(y) * np.cos(x)
    uy = lambda x, y: u_profile(y) * np.sin(x)
    p = lambda x, y: np.zeros_like(np.asarray(x, dtype=float)) + float(p_const)
    phi = lambda x, y: np.exp(y) * np.sin(x) + float(p_const) / float(g)
    grad_phi = lambda x, y: _stack_vec(
        lambda xv, yv: np.exp(yv) * np.cos(xv),
        lambda xv, yv: np.exp(yv) * np.sin(xv),
    )(x, y)
    f1 = lambda x, y: (
        -0.5 * float(g) + (float(nu) * float(K) - (float(friction_alpha) * float(g)) / (2.0 * float(nu))) * np.asarray(y, dtype=float)
    ) * np.cos(x)
    f2 = lambda x, y: (
        (float(friction_alpha) * float(g)) / (2.0 * float(nu))
        - 2.0 * float(nu) * float(K)
        - 0.5 * float(g) * np.asarray(y, dtype=float)
        + (0.5 * float(nu) * float(K) - (float(friction_alpha) * float(g)) / (4.0 * float(nu))) * (np.asarray(y, dtype=float) ** 2)
    ) * np.sin(x)
    return {
        "ux": ux,
        "uy": uy,
        "p": p,
        "phi": phi,
        "grad_phi": grad_phi,
        "f_vec": _stack_vec(f1, f2),
    }


def _phase_field(*, eps: float):
    eps_val = float(eps)

    def c(x, y):
        return 0.5 * (1.0 + np.tanh(np.asarray(y, dtype=float) / eps_val))

    def dc_dy(x, y):
        yy = np.asarray(y, dtype=float) / eps_val
        sech2 = 1.0 / np.cosh(yy) ** 2
        return 0.5 * sech2 / eps_val

    grad_c = lambda x, y: _stack_vec(lambda xv, yv: np.zeros_like(np.asarray(xv, dtype=float)), dc_dy)(x, y)
    abs_grad_c = lambda x, y: np.abs(dc_dy(x, y))
    return c, grad_c, abs_grad_c


def _eval_scalar_at_point(dh: DofHandler, mesh: Mesh, field_name: str, values: np.ndarray, point: tuple[float, float]) -> float:
    from pycutfem.fem import transform

    xy = np.asarray(point, dtype=float)
    for elem in mesh.elements_list:
        verts = mesh.nodes_x_y_pos[list(elem.nodes)]
        if not (
            verts[:, 0].min() - 1.0e-12 <= xy[0] <= verts[:, 0].max() + 1.0e-12
            and verts[:, 1].min() - 1.0e-12 <= xy[1] <= verts[:, 1].max() + 1.0e-12
        ):
            continue
        try:
            xi, eta = transform.inverse_mapping(mesh, elem.id, xy)
        except Exception:
            continue
        if not (-1.0001 <= xi <= 1.0001 and -1.0001 <= eta <= 1.0001):
            continue
        phi = dh.mixed_element.basis(field_name, float(xi), float(eta))[dh.mixed_element.slice(field_name)]
        gdofs = np.asarray(dh.element_maps[field_name][elem.id], dtype=int)
        return float(np.asarray(phi, dtype=float) @ np.asarray(values[gdofs], dtype=float))
    raise RuntimeError(f"Failed to locate point {point} in the Stoter mesh.")


@dataclass(frozen=True)
class CaseResult:
    nx: int
    ny: int
    eps: float
    rmse_u_stokes: float
    max_u_stokes: float
    rmse_p_stokes: float
    rmse_phi_darcy: float
    max_phi_darcy: float


def solve_case(*, nx: int, ny: int, eps_over_h: float, nu: float, K: float, friction_alpha: float, g: float, p_const: float, backend: str) -> CaseResult:
    Lx = float(np.pi)
    Ly = 2.0
    nodes, elems, _, corners = structured_quad(Lx, Ly, nx=int(nx), ny=int(ny), poly_order=2, offset=(0.0, -1.0))
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=2)
    _tag_boundaries(mesh, Lx=Lx)

    me = MixedElement(mesh, field_specs={"ux": 2, "uy": 2, "p": 1, "phi_D": 2})
    dh = DofHandler(me, method="cg")
    dh.tag_dof_by_locator("pressure_pin", "p", lambda x, y: abs(x - 0.0) <= 1.0e-12 and abs(y - 1.0) <= 1.0e-12)

    V = FunctionSpace("V", ["ux", "uy"], dim=1)
    u = VectorTrialFunction(space=V, dof_handler=dh)
    v = VectorTestFunction(space=V, dof_handler=dh)
    p = TrialFunction("p", dof_handler=dh)
    q = TestFunction("p", dof_handler=dh)
    phi_D = TrialFunction("phi_D", dof_handler=dh)
    psi = TestFunction("phi_D", dof_handler=dh)

    h = min(float(Lx) / float(nx), float(Ly) / float(ny))
    eps = float(eps_over_h) * h
    exact = _stoter_exact_fields(nu=nu, K=K, friction_alpha=friction_alpha, g=g, p_const=p_const)
    c_fun, grad_c_fun, abs_grad_c_fun = _phase_field(eps=eps)

    c_expr = Analytic(c_fun, degree=8)
    one_minus_c_expr = Analytic(lambda x, y: 1.0 - c_fun(x, y), degree=8)
    grad_c_expr = Analytic(grad_c_fun, dim=1, degree=8)
    abs_grad_c_expr = Analytic(abs_grad_c_fun, degree=8)
    f_expr = Analytic(exact["f_vec"], dim=1, degree=8)

    a = (
        (Constant(2.0 * float(nu)) * inner(_eps(u), _eps(v)) - p * div(v) + q * div(u)) * c_expr
        + Constant(float(K)) * dot(grad(phi_D), grad(psi)) * one_minus_c_expr
        + psi * dot(u, grad_c_expr)
        - Constant(float(g)) * phi_D * dot(v, grad_c_expr)
        + Constant(float(friction_alpha / math.sqrt(K))) * u[0] * v[0] * abs_grad_c_expr
    ) * dx()
    L = (dot(f_expr, v) * c_expr) * dx()

    bcs = [
        BoundaryCondition("ux", "dirichlet", "stokes_top", lambda x, y: float(exact["ux"](x, y))),
        BoundaryCondition("uy", "dirichlet", "stokes_top", lambda x, y: float(exact["uy"](x, y))),
        BoundaryCondition("ux", "dirichlet", "stokes_left", lambda x, y: float(exact["ux"](x, y))),
        BoundaryCondition("uy", "dirichlet", "stokes_left", lambda x, y: float(exact["uy"](x, y))),
        BoundaryCondition("ux", "dirichlet", "stokes_right", lambda x, y: float(exact["ux"](x, y))),
        BoundaryCondition("uy", "dirichlet", "stokes_right", lambda x, y: float(exact["uy"](x, y))),
        BoundaryCondition("phi_D", "dirichlet", "darcy_bottom", lambda x, y: float(exact["phi"](x, y))),
        BoundaryCondition("phi_D", "dirichlet", "darcy_left", lambda x, y: float(exact["phi"](x, y))),
        BoundaryCondition("phi_D", "dirichlet", "darcy_right", lambda x, y: float(exact["phi"](x, y))),
        BoundaryCondition("p", "dirichlet", "pressure_pin", lambda x, y: float(p_const)),
    ]

    A, b = assemble_form(Equation(a, L), dof_handler=dh, bcs=bcs, quad_order=8, backend=backend)
    sol = np.asarray(spla.spsolve(A.tocsc(), b), dtype=float).ravel()

    ux_slice = np.asarray(dh.get_field_slice("ux"), dtype=int)
    uy_slice = np.asarray(dh.get_field_slice("uy"), dtype=int)
    p_slice = np.asarray(dh.get_field_slice("p"), dtype=int)
    phi_slice = np.asarray(dh.get_field_slice("phi_D"), dtype=int)

    stokes_points = [(float(x), float(y)) for x in np.linspace(0.0, Lx, 81) for y in np.linspace(0.05, 0.95, 33)]
    darcy_points = [(float(x), float(y)) for x in np.linspace(0.0, Lx, 81) for y in np.linspace(-0.95, -0.05, 33)]

    u_err_sq = []
    u_max = 0.0
    p_err_sq = []
    for x, y in stokes_points:
        ux_h = _eval_scalar_at_point(dh, mesh, "ux", sol, (x, y))
        uy_h = _eval_scalar_at_point(dh, mesh, "uy", sol, (x, y))
        p_h = _eval_scalar_at_point(dh, mesh, "p", sol, (x, y))
        ux_ex = float(exact["ux"](x, y))
        uy_ex = float(exact["uy"](x, y))
        p_ex = float(exact["p"](x, y))
        u_err_sq.append((ux_h - ux_ex) ** 2 + (uy_h - uy_ex) ** 2)
        p_err_sq.append((p_h - p_ex) ** 2)
        u_max = max(u_max, math.sqrt(ux_ex * ux_ex + uy_ex * uy_ex))

    phi_err_sq = []
    phi_max = 0.0
    for x, y in darcy_points:
        phi_h = _eval_scalar_at_point(dh, mesh, "phi_D", sol, (x, y))
        phi_ex = float(exact["phi"](x, y))
        phi_err_sq.append((phi_h - phi_ex) ** 2)
        phi_max = max(phi_max, abs(phi_ex))

    return CaseResult(
        nx=int(nx),
        ny=int(ny),
        eps=float(eps),
        rmse_u_stokes=float(math.sqrt(float(np.mean(np.asarray(u_err_sq, dtype=float))))),
        max_u_stokes=float(u_max),
        rmse_p_stokes=float(math.sqrt(float(np.mean(np.asarray(p_err_sq, dtype=float))))),
        rmse_phi_darcy=float(math.sqrt(float(np.mean(np.asarray(phi_err_sq, dtype=float))))),
        max_phi_darcy=float(phi_max),
    )


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one resolution entry.")
    return values


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=str, default="out/benchmark7_stoter_stokes_darcy")
    ap.add_argument("--nx-list", type=str, default="16,32,64")
    ap.add_argument("--backend", type=str, default="python", choices=("python", "jit", "cpp"))
    ap.add_argument("--eps-over-h", type=float, default=1.0)
    ap.add_argument("--nu", type=float, default=1.0)
    ap.add_argument("--K", type=float, default=1.0)
    ap.add_argument("--friction-alpha", type=float, default=1.0)
    ap.add_argument("--g", type=float, default=9.81)
    ap.add_argument("--p-const", type=float, default=1.0)
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for nx in _parse_int_list(args.nx_list):
        rows.append(
            solve_case(
                nx=int(nx),
                ny=2 * int(nx),
                eps_over_h=float(args.eps_over_h),
                nu=float(args.nu),
                K=float(args.K),
                friction_alpha=float(args.friction_alpha),
                g=float(args.g),
                p_const=float(args.p_const),
                backend=str(args.backend),
            )
        )

    summary = {
        "benchmark": "stoter_stokes_darcy_straight_interface",
        "backend": str(args.backend),
        "eps_over_h": float(args.eps_over_h),
        "nu": float(args.nu),
        "K": float(args.K),
        "friction_alpha": float(args.friction_alpha),
        "g": float(args.g),
        "p_const": float(args.p_const),
        "cases": [row.__dict__ for row in rows],
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
