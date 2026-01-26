#!/usr/bin/env python3

from __future__ import annotations

import argparse
from math import pi

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sympy as sp_sym

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.nonmatching import build_nonmatching_interface
from pycutfem.nonmatching.diagnostics import stokes_traction_mismatch_L2, stokes_velocity_jump_L2
from pycutfem.nonmatching.nitsche import assemble_stokes_nitsche_interface_matrix
from pycutfem.nonmatching.norms import scalar_L2_error
from pycutfem.nonmatching.system import apply_dirichlet_data, coupled_dirichlet_data
from pycutfem.ufl.analytic import Analytic, x, y
from pycutfem.ufl.expressions import (
    Constant,
    TestFunction,
    TrialFunction,
    VectorTestFunction,
    VectorTrialFunction,
    div,
    dot,
    grad,
    inner,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad, structured_triangles


def _make_submesh(*, element: str, nx: int, ny: int, offset_x: float) -> Mesh:
    if element == "quad":
        nodes, elems, edges, corners = structured_quad(0.5, 1.0, nx=nx, ny=ny, poly_order=2, offset=(offset_x, 0.0))
        return Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=2)
    if element == "tri":
        nodes, elems, edges, corners = structured_triangles(
            0.5, 1.0, nx_quads=nx, ny_quads=ny, poly_order=2, offset=(offset_x, 0.0)
        )
        return Mesh(nodes, elems, edges, corners, element_type="tri", poly_order=2)
    raise ValueError("element must be 'quad' or 'tri'")


def _char_h(mesh: Mesh) -> float:
    a = np.asarray(mesh.areas_list, dtype=float)
    a = a[a > 0.0]
    return float(np.sqrt(a.max())) if a.size else 1.0


def _rate(h0: float, e0: float, h1: float, e1: float) -> float:
    if e1 <= 0.0 or e0 <= 0.0:
        return float("nan")
    return float(np.log(e0 / e1) / np.log(h0 / h1))


def solve_level(
    *,
    element: str,
    ny_neg: int,
    ny_pos: int,
    mu: float,
    gamma: float,
    backend_volume: str,
    backend_interface: str,
) -> dict[str, float]:
    nx_neg = max(2, int(round(0.5 * ny_neg)))
    nx_pos = max(2, int(round(0.5 * ny_pos)))

    mesh_neg = _make_submesh(element=element, nx=nx_neg, ny=ny_neg, offset_x=0.0)
    mesh_pos = _make_submesh(element=element, nx=nx_pos, ny=ny_pos, offset_x=0.5)

    mesh_neg.tag_boundary_edges(
        {"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True}
    )
    mesh_pos.tag_boundary_edges(
        {"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True}
    )

    fields = {"ux": 2, "uy": 2, "p": 1}
    dh_neg = DofHandler(MixedElement(mesh_neg, fields), method="cg")
    dh_pos = DofHandler(MixedElement(mesh_pos, fields), method="cg")

    # Pressure gauge: pin one pressure DOF per submesh.
    dh_neg.tag_dof_by_locator("pressure_pin", "p", lambda xx, yy: abs(xx - 0.0) < 1e-12 and abs(yy - 0.0) < 1e-12)
    dh_pos.tag_dof_by_locator("pressure_pin", "p", lambda xx, yy: abs(xx - 1.0) < 1e-12 and abs(yy - 0.0) < 1e-12)

    # MMS (Test 4): divergence-free velocity from streamfunction.
    psi = sp_sym.sin(sp_sym.pi * x) * sp_sym.sin(sp_sym.pi * y)
    ux_expr = sp_sym.diff(psi, y)
    uy_expr = -sp_sym.diff(psi, x)
    p_expr = sp_sym.sin(2 * sp_sym.pi * x) * sp_sym.sin(2 * sp_sym.pi * y)
    f1_expr = 2 * mu * sp_sym.pi**3 * sp_sym.sin(sp_sym.pi * x) * sp_sym.cos(sp_sym.pi * y) + 2 * sp_sym.pi * sp_sym.cos(2 * sp_sym.pi * x) * sp_sym.sin(2 * sp_sym.pi * y)
    f2_expr = -2 * mu * sp_sym.pi**3 * sp_sym.cos(sp_sym.pi * x) * sp_sym.sin(sp_sym.pi * y) + 2 * sp_sym.pi * sp_sym.sin(2 * sp_sym.pi * x) * sp_sym.cos(2 * sp_sym.pi * y)

    ux_fun = sp_sym.lambdify((x, y), ux_expr, "numpy")
    uy_fun = sp_sym.lambdify((x, y), uy_expr, "numpy")
    p_fun = sp_sym.lambdify((x, y), p_expr, "numpy")
    f1_fun = sp_sym.lambdify((x, y), f1_expr, "numpy")
    f2_fun = sp_sym.lambdify((x, y), f2_expr, "numpy")
    stack_vec = lambda fx, fy: lambda xv, yv: np.stack([fx(xv, yv), fy(xv, yv)], axis=-1)
    f_vec = Analytic(stack_vec(f1_fun, f2_fun), dim=1, degree=8)

    vel_space = FunctionSpace("V", ["ux", "uy"], dim=1)
    u = VectorTrialFunction(vel_space, dof_handler=dh_pos)  # dof_handler overridden in assemble_form
    v = VectorTestFunction(vel_space, dof_handler=dh_pos)
    p = TrialFunction("p", dof_handler=dh_pos)
    q = TestFunction("p", dof_handler=dh_pos)

    def eps(w):
        return 0.5 * (grad(w) + grad(w).T)

    a = (Constant(2.0 * mu) * inner(eps(u), eps(v)) - p * div(v) + q * div(u)) * dx()
    L = dot(f_vec, v) * dx()
    qdeg = 8

    def assemble_side(dh: DofHandler):
        return assemble_form(Equation(a, L), dof_handler=dh, bcs=[], quad_order=qdeg, backend=backend_volume)

    K_pos, F_pos = assemble_side(dh_pos)
    K_neg, F_neg = assemble_side(dh_neg)

    interface = build_nonmatching_interface(
        mesh_neg=mesh_neg, mesh_pos=mesh_pos, neg_edges="interface", pos_edges="interface"
    )
    K_if = assemble_stokes_nitsche_interface_matrix(
        interface=interface,
        dh_neg=dh_neg,
        dh_pos=dh_pos,
        mu_neg=float(mu),
        mu_pos=float(mu),
        gamma=float(gamma),
        backend=backend_interface,
    )

    n_pos = int(dh_pos.total_dofs)
    n_neg = int(dh_neg.total_dofs)
    K = sp.block_diag([K_pos.tocsr(), K_neg.tocsr()], format="csr") + K_if
    F = np.concatenate([np.asarray(F_pos, float), np.asarray(F_neg, float)])

    bcs_pos = [
        BoundaryCondition("ux", "dirichlet", "boundary", lambda xx, yy: float(ux_fun(xx, yy))),
        BoundaryCondition("uy", "dirichlet", "boundary", lambda xx, yy: float(uy_fun(xx, yy))),
        BoundaryCondition("p", "dirichlet", "pressure_pin", lambda xx, yy: float(p_fun(xx, yy))),
    ]
    bcs_neg = [
        BoundaryCondition("ux", "dirichlet", "boundary", lambda xx, yy: float(ux_fun(xx, yy))),
        BoundaryCondition("uy", "dirichlet", "boundary", lambda xx, yy: float(uy_fun(xx, yy))),
        BoundaryCondition("p", "dirichlet", "pressure_pin", lambda xx, yy: float(p_fun(xx, yy))),
    ]
    bc_data = coupled_dirichlet_data(dh_pos=dh_pos, bcs_pos=bcs_pos, dh_neg=dh_neg, bcs_neg=bcs_neg, neg_offset=n_pos)
    K_bc, F_bc = apply_dirichlet_data(K, F, bc_data)

    sol = spla.spsolve(K_bc.tocsc(), F_bc)
    U_pos = sol[:n_pos]
    U_neg = sol[n_pos : n_pos + n_neg]

    # L2 errors (velocity, pressure)
    err_ux_pos = scalar_L2_error(dh=dh_pos, uh=U_pos, u_exact=lambda xx, yy: float(ux_fun(xx, yy)), field="ux")
    err_uy_pos = scalar_L2_error(dh=dh_pos, uh=U_pos, u_exact=lambda xx, yy: float(uy_fun(xx, yy)), field="uy")
    err_ux_neg = scalar_L2_error(dh=dh_neg, uh=U_neg, u_exact=lambda xx, yy: float(ux_fun(xx, yy)), field="ux")
    err_uy_neg = scalar_L2_error(dh=dh_neg, uh=U_neg, u_exact=lambda xx, yy: float(uy_fun(xx, yy)), field="uy")
    err_u = float(np.sqrt(err_ux_pos**2 + err_uy_pos**2 + err_ux_neg**2 + err_uy_neg**2))

    err_p_pos = scalar_L2_error(dh=dh_pos, uh=U_pos, u_exact=lambda xx, yy: float(p_fun(xx, yy)), field="p")
    err_p_neg = scalar_L2_error(dh=dh_neg, uh=U_neg, u_exact=lambda xx, yy: float(p_fun(xx, yy)), field="p")
    err_p = float(np.sqrt(err_p_pos**2 + err_p_neg**2))

    jump_u = stokes_velocity_jump_L2(interface=interface, dh_neg=dh_neg, U_neg=U_neg, dh_pos=dh_pos, U_pos=U_pos)
    jump_t = stokes_traction_mismatch_L2(
        interface=interface,
        dh_neg=dh_neg,
        U_neg=U_neg,
        dh_pos=dh_pos,
        U_pos=U_pos,
        mu_neg=float(mu),
        mu_pos=float(mu),
    )

    h = max(_char_h(mesh_neg), _char_h(mesh_pos))
    return {"h": float(h), "u": err_u, "p": err_p, "jump_u": float(jump_u), "jump_t": float(jump_t)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--element", choices=("quad", "tri"), default="quad")
    p.add_argument("--n0", type=int, default=6, help="base ny on the negative side")
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=40.0)
    p.add_argument("--backend-volume", choices=("python", "jit", "cpp"), default="jit")
    p.add_argument("--backend-interface", choices=("python", "jit", "cpp"), default="python")
    args = p.parse_args()

    rows: list[dict[str, float]] = []
    for lev in range(int(args.levels)):
        ny_neg = int(args.n0) * (2**lev)
        ny_pos = ny_neg + 1
        rows.append(
            solve_level(
                element=str(args.element),
                ny_neg=ny_neg,
                ny_pos=ny_pos,
                mu=float(args.mu),
                gamma=float(args.gamma),
                backend_volume=str(args.backend_volume),
                backend_interface=str(args.backend_interface),
            )
        )

    print("\nNon-matching Stokes–Stokes (Nitsche) MMS: interface x=0.5")
    print(
        f"element={args.element} volume={args.backend_volume} interface={args.backend_interface} mu={args.mu} gamma={args.gamma}"
    )
    print("\n  i       h           ||u||_L2       rate      ||p||_L2       rate      ||[u]||_G      ||[sigma n]||_G")
    for i, r in enumerate(rows):
        if i == 0:
            print(
                f"{i:3d}  {r['h']:.3e}  {r['u']:.3e}    {'-':>6}  {r['p']:.3e}    {'-':>6}  {r['jump_u']:.3e}  {r['jump_t']:.3e}"
            )
            continue
        r0 = rows[i - 1]
        print(
            f"{i:3d}  {r['h']:.3e}  {r['u']:.3e}  {_rate(r0['h'], r0['u'], r['h'], r['u']):6.2f}  "
            f"{r['p']:.3e}  {_rate(r0['h'], r0['p'], r['h'], r['p']):6.2f}  {r['jump_u']:.3e}  {r['jump_t']:.3e}"
        )


if __name__ == "__main__":
    main()

