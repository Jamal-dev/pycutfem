#!/usr/bin/env python3

from __future__ import annotations

import argparse
from math import pi

import numpy as np
import scipy.sparse.linalg as spla
import sympy as sp_sym

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.nonmatching import (
    build_nonmatching_interface,
    poisson_flux_mismatch_L2,
    scalar_H1_semi_error,
    scalar_L2_error,
    scalar_jump_L2,
)
from pycutfem.nonmatching.mortar import assemble_mortar_saddle_matrix, assemble_poisson_mortar_coupling
from pycutfem.nonmatching.system import apply_dirichlet_data, coupled_dirichlet_data
from pycutfem.ufl.analytic import Analytic, x, y
from pycutfem.ufl.expressions import Constant, TestFunction, TrialFunction, grad, inner
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad, structured_triangles


def _make_submesh(*, element: str, degree: int, nx: int, ny: int, offset_x: float) -> Mesh:
    if element == "quad":
        nodes, elems, edges, corners = structured_quad(
            0.5, 1.0, nx=nx, ny=ny, poly_order=degree, offset=(offset_x, 0.0)
        )
        return Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=degree)
    if element == "tri":
        nodes, elems, edges, corners = structured_triangles(
            0.5, 1.0, nx_quads=nx, ny_quads=ny, poly_order=degree, offset=(offset_x, 0.0)
        )
        return Mesh(nodes, elems, edges, corners, element_type="tri", poly_order=degree)
    raise ValueError("element must be 'quad' or 'tri'")


def _char_h(mesh: Mesh) -> float:
    a = np.asarray(mesh.areas_list, dtype=float)
    a = a[a > 0.0]
    return float(np.sqrt(a.max())) if a.size else 1.0


def solve_level(
    *,
    element: str,
    degree: int,
    ny_neg: int,
    ny_pos: int,
    k_neg: float,
    k_pos: float,
    backend_volume: str,
    backend_interface: str,
) -> dict[str, float]:
    nx_neg = max(2, int(round(0.5 * ny_neg)))
    nx_pos = max(2, int(round(0.5 * ny_pos)))

    mesh_neg = _make_submesh(element=element, degree=degree, nx=nx_neg, ny=ny_neg, offset_x=0.0)
    mesh_pos = _make_submesh(element=element, degree=degree, nx=nx_pos, ny=ny_pos, offset_x=0.5)

    mesh_neg.tag_boundary_edges(
        {"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True}
    )
    mesh_pos.tag_boundary_edges(
        {"interface": lambda xx, yy: abs(xx - 0.5) < 1e-12, "boundary": lambda xx, yy: True}
    )

    dh_neg = DofHandler(MixedElement(mesh_neg, {"u": int(degree)}), method="cg")
    dh_pos = DofHandler(MixedElement(mesh_pos, {"u": int(degree)}), method="cg")

    # MMS (Test 3): coefficient jump with matching physics.
    u_neg_expr = (x - sp_sym.Rational(1, 2)) * sp_sym.sin(sp_sym.pi * y)
    u_pos_expr = sp_sym.Rational(1, 10) * (x - sp_sym.Rational(1, 2)) * sp_sym.sin(sp_sym.pi * y)
    f_expr = (sp_sym.pi**2) * (x - sp_sym.Rational(1, 2)) * sp_sym.sin(sp_sym.pi * y)

    u_neg_fun = sp_sym.lambdify((x, y), u_neg_expr, "numpy")
    u_pos_fun = sp_sym.lambdify((x, y), u_pos_expr, "numpy")

    def grad_u_neg(xx: float, yy: float) -> tuple[float, float]:
        return (float(np.sin(pi * yy)), float((xx - 0.5) * pi * np.cos(pi * yy)))

    def grad_u_pos(xx: float, yy: float) -> tuple[float, float]:
        return (0.1 * float(np.sin(pi * yy)), 0.1 * float((xx - 0.5) * pi * np.cos(pi * yy)))

    f = Analytic(f_expr, degree=max(4, 2 * degree + 2))

    def assemble_side(dh: DofHandler, k: float):
        u = TrialFunction(name="u_trial", field_name="u", dof_handler=dh)
        v = TestFunction(name="v_test", field_name="u", dof_handler=dh)
        a = Constant(float(k)) * inner(grad(u), grad(v)) * dx()
        L = f * v * dx()
        q = int(2 * degree + 2)
        return assemble_form(Equation(a, L), dof_handler=dh, bcs=[], quad_order=q, backend=backend_volume)

    K_pos, F_pos = assemble_side(dh_pos, k_pos)
    K_neg, F_neg = assemble_side(dh_neg, k_neg)

    interface = build_nonmatching_interface(
        mesh_neg=mesh_neg, mesh_pos=mesh_pos, neg_edges="interface", pos_edges="interface"
    )
    coupling = assemble_poisson_mortar_coupling(
        interface=interface, dh_neg=dh_neg, dh_pos=dh_pos, field="u", backend=backend_interface
    )
    K = assemble_mortar_saddle_matrix(K_pos=K_pos, K_neg=K_neg, coupling=coupling)

    n_pos = int(dh_pos.total_dofs)
    n_neg = int(dh_neg.total_dofs)
    n_lam = int(coupling.n_lambda)
    F = np.concatenate([np.asarray(F_pos, float), np.asarray(F_neg, float), np.zeros(n_lam, dtype=float)])

    bcs_pos = [BoundaryCondition("u", "dirichlet", "boundary", lambda xx, yy: float(u_pos_fun(xx, yy)))]
    bcs_neg = [BoundaryCondition("u", "dirichlet", "boundary", lambda xx, yy: float(u_neg_fun(xx, yy)))]
    bc_data = coupled_dirichlet_data(
        dh_pos=dh_pos, bcs_pos=bcs_pos, dh_neg=dh_neg, bcs_neg=bcs_neg, neg_offset=n_pos
    )
    K_bc, F_bc = apply_dirichlet_data(K, F, bc_data)

    sol = spla.spsolve(K_bc.tocsc(), F_bc)
    U_pos = sol[:n_pos]
    U_neg = sol[n_pos : n_pos + n_neg]

    errL2_pos = scalar_L2_error(dh=dh_pos, uh=U_pos, u_exact=lambda xx, yy: float(u_pos_fun(xx, yy)), field="u")
    errL2_neg = scalar_L2_error(dh=dh_neg, uh=U_neg, u_exact=lambda xx, yy: float(u_neg_fun(xx, yy)), field="u")
    errH1_pos = scalar_H1_semi_error(dh=dh_pos, uh=U_pos, grad_u_exact=grad_u_pos, field="u")
    errH1_neg = scalar_H1_semi_error(dh=dh_neg, uh=U_neg, grad_u_exact=grad_u_neg, field="u")
    errL2 = float(np.sqrt(errL2_pos**2 + errL2_neg**2))
    errH1 = float(np.sqrt(errH1_pos**2 + errH1_neg**2))

    jump = scalar_jump_L2(interface=interface, dh_neg=dh_neg, u_neg=U_neg, dh_pos=dh_pos, u_pos=U_pos, field="u")
    flux = poisson_flux_mismatch_L2(
        interface=interface,
        dh_neg=dh_neg,
        u_neg=U_neg,
        dh_pos=dh_pos,
        u_pos=U_pos,
        field="u",
        k_neg=k_neg,
        k_pos=k_pos,
    )

    h = max(_char_h(mesh_neg), _char_h(mesh_pos))
    return {"h": float(h), "L2": errL2, "H1": errH1, "jump": float(jump), "flux": float(flux), "n_lambda": float(n_lam)}


def _rate(h0: float, e0: float, h1: float, e1: float) -> float:
    if e1 <= 0.0 or e0 <= 0.0:
        return float("nan")
    return float(np.log(e0 / e1) / np.log(h0 / h1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--element", choices=("quad", "tri"), default="quad")
    p.add_argument("--degree", type=int, default=1)
    p.add_argument("--n0", type=int, default=6, help="base ny on the negative side")
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--k-neg", type=float, default=1.0)
    p.add_argument("--k-pos", type=float, default=10.0)
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
                degree=int(args.degree),
                ny_neg=ny_neg,
                ny_pos=ny_pos,
                k_neg=float(args.k_neg),
                k_pos=float(args.k_pos),
                backend_volume=str(args.backend_volume),
                backend_interface=str(args.backend_interface),
            )
        )

    print("\nNon-matching Poisson (Mortar) MMS: k-jump, interface x=0.5")
    print(
        f"element={args.element} degree={args.degree} volume={args.backend_volume} interface={args.backend_interface}"
    )
    print("\n  i       h           L2             rate      H1             rate      ||[u]||_G      ||flux||_G    n_lambda")
    for i, r in enumerate(rows):
        if i == 0:
            print(
                f"{i:3d}  {r['h']:.3e}  {r['L2']:.3e}    {'-':>6}  {r['H1']:.3e}    {'-':>6}  {r['jump']:.3e}  {r['flux']:.3e}  {int(r['n_lambda']):7d}"
            )
            continue
        r0 = rows[i - 1]
        print(
            f"{i:3d}  {r['h']:.3e}  {r['L2']:.3e}  {_rate(r0['h'], r0['L2'], r['h'], r['L2']):6.2f}  "
            f"{r['H1']:.3e}  {_rate(r0['h'], r0['H1'], r['h'], r['H1']):6.2f}  {r['jump']:.3e}  {r['flux']:.3e}  {int(r['n_lambda']):7d}"
        )


if __name__ == "__main__":
    main()

