#!/usr/bin/env python3
"""
Compare PyCutFEM ghost/interface integrals against the deal.II step-85 formulas.

Context
-------
deal.II v9.7.0 `examples/step-85` implements CutFEM for a scalar Laplace problem
on a unit disk (immersed in a Cartesian background mesh). Two terms are
particularly relevant for debugging:

1) Ghost penalty (facet-based, "deal.II-style")
   In `step-85.cc` (assemble_system), the ghost penalty uses FEInterfaceValues:

     a_ghost(u,v) = γ * h_F ∫_{F_h} (n · [∇u]) (n · [∇v]) ds

   with γ = 0.5 and h_F = cell->minimum_vertex_distance(). The source multiplies
   by 1/2 because it traverses each interior face twice.

   Key: This is a *facet* integral that penalizes jumps of the *normal
   derivative*. It is non-zero for CG spaces (since ∇u is discontinuous).

2) Symmetric Nitsche interface term (immersed boundary Γ)
     a_Γ(u,v) = ∫_Γ (-(n·∇u) v -(n·∇v) u + β/h u v) ds
   with β = 5*(k+1)*k for FE degree k.

PyCutFEM measures
-----------------
- `dGhost`      : facet-based ghost/cut-skeleton integrals (matches deal.II style)
- `dFacetPatch` : *two-element volume patch* integrals with polynomial extension
                 (NGSolve/AgFEM-style). For CG spaces, `jump(u)` on `dFacetPatch`
                 is generally non-zero, unlike `jump(u)` on facets.

This script assembles energies uᵀ K u for:
  - `jump(u)` on `dGhost`        (≈0 for CG)
  - `jump(grad(u),n)` on `dGhost` (deal.II step-85 ghost penalty)
  - `jump(u)` on `dFacetPatch`   (patch/extension stabilization)
  - the Nitsche interface bilinear form on `dInterface`

It also provides an optional manual quadrature check for the deal.II-style ghost
penalty on Q1 structured quads.
"""

from __future__ import annotations

import argparse
import os
from typing import Callable

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import CircleLevelSet, PiecewiseLinearLevelSet
from pycutfem.core.mesh import Mesh as PCMesh
from pycutfem.fem import transform
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.integration.quadrature import line_quadrature
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    FacetNormal,
    TestFunction,
    TrialFunction,
    VectorTestFunction,
    VectorTrialFunction,
    dot,
    grad,
    jump,
)
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dFacetPatch, dGhost, dInterface
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def _default_backend() -> str:
    backend = (os.getenv("BACKEND") or os.getenv("PYCUTFEM_BACKEND") or "python").strip().lower()
    return backend or "python"


def assemble_energy(form, dh: DofHandler, u_vec: np.ndarray, *, backend: str) -> float:
    K, _ = assemble_form(Equation(form, None), dof_handler=dh, bcs=[], backend=backend)
    return float(u_vec @ (K @ u_vec))


def _fill_field(
    vec: np.ndarray,
    dh: DofHandler,
    field: str,
    fn_xy: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    ids = np.asarray(dh.get_field_slice(field), dtype=int)
    xy = np.asarray(dh.get_dof_coords(field), dtype=float)
    x = xy[:, 0]
    y = xy[:, 1]
    vec[ids] = np.asarray(fn_xy(x, y), dtype=float).reshape(-1)


def manual_ghost_energy_q1(
    *,
    mesh: PCMesh,
    dh: DofHandler,
    me: MixedElement,
    u_vec: np.ndarray,
    ghost_edges,
    gamma: float,
    q: int,
    h_mode: str,
) -> float:
    """
    Manual implementation of the deal.II step-85 ghost penalty energy for a scalar field:

      E = γ * h_F ∑_{F∈F_h} ∫_F (n · [∇u])² ds

    Notes
    -----
    - This checks the *mathematical* integral, not the plus/minus orientation.
      The energy is invariant under normal flips.
    - Only intended for affine Q1 geometry (poly_order=1 structured quads).
    """
    if h_mode not in {"left", "min", "avg"}:
        raise ValueError("h_mode must be 'left', 'min', or 'avg'.")

    E = 0.0
    for edge_id in ghost_edges.to_indices():
        e = mesh.edges_list[int(edge_id)]
        if e.right is None:
            continue
        left = int(e.left)
        right = int(e.right)

        p0, p1 = mesh.nodes_x_y_pos[list(e.nodes)]
        qpts, qwts = line_quadrature(p0, p1, order=int(q))

        t = np.asarray(p1 - p0, dtype=float)
        tn = float(np.linalg.norm(t))
        if tn <= 1e-30:
            continue
        normal = np.array([t[1], -t[0]], dtype=float) / tn

        h_left = float(mesh.element_char_length(left))
        h_right = float(mesh.element_char_length(right))
        if h_mode == "left":
            h_F = h_left
        elif h_mode == "min":
            h_F = min(h_left, h_right)
        else:
            h_F = 0.5 * (h_left + h_right)

        dofs_left = np.asarray(dh.get_elemental_dofs(left), dtype=int)
        dofs_right = np.asarray(dh.get_elemental_dofs(right), dtype=int)
        u_left = u_vec[dofs_left]
        u_right = u_vec[dofs_right]

        for xq, w in zip(qpts, qwts):
            xi_l, eta_l = transform.inverse_mapping(mesh, left, np.asarray(xq, dtype=float))
            xi_r, eta_r = transform.inverse_mapping(mesh, right, np.asarray(xq, dtype=float))

            J_l = transform.jacobian(mesh, left, (float(xi_l), float(eta_l)))
            J_r = transform.jacobian(mesh, right, (float(xi_r), float(eta_r)))
            Ji_l = np.linalg.inv(J_l)
            Ji_r = np.linalg.inv(J_r)

            # MixedElement.*basis are union-sized; other fields contribute 0 here.
            G_l = me.grad_basis("u", float(xi_l), float(eta_l)) @ Ji_l  # (n_loc,2)
            G_r = me.grad_basis("u", float(xi_r), float(eta_r)) @ Ji_r

            grad_u_l = u_left @ G_l  # (2,)
            grad_u_r = u_right @ G_r
            jump_dn = float(normal @ (grad_u_l - grad_u_r))
            E += float(gamma) * h_F * (jump_dn * jump_dn) * float(w)

    return float(E)


def _print_abs_rel(label: str, a: float, b: float) -> None:
    abs_err = float(abs(a - b))
    scale = max(float(abs(a)), float(abs(b)), 1.0)
    rel_err = abs_err / scale
    print(f"  {label}: {a:+.12e}  ref={b:+.12e}  |diff|={abs_err:.3e}  rel={rel_err:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("python", "jit", "cpp"), default=_default_backend())
    parser.add_argument("--nx", type=int, default=4, help="Cells per direction (step-85 cycle0 ~ 4)")
    parser.add_argument("--L", type=float, default=2.42, help="Domain size (step-85 uses [-1.21,1.21] so L=2.42)")
    parser.add_argument("--R", type=float, default=1.0, help="Circle radius for the level set (unit disk)")
    parser.add_argument("--order", type=int, default=1, help="Scalar FE order (step-85 uses 1)")
    parser.add_argument("--q-ghost", type=int, default=2, help="Facet quadrature for dGhost (step-85 uses k+1=2)")
    parser.add_argument("--q-patch", type=int, default=8, help="Volume quadrature for dFacetPatch")
    parser.add_argument("--q-interface", type=int, default=10, help="Interface quadrature for dInterface")
    parser.add_argument("--no-manual-check", action="store_true", help="Skip the manual dGhost check")
    parser.add_argument(
        "--manual-h",
        choices=("left", "min", "avg"),
        default="left",
        help="Manual check: h_F choice for non-uniform meshes (step-85 effective is 'avg' due to double traversal).",
    )
    parser.add_argument("--mixed", action="store_true", help="Also run a mixed-field (ux,uy,dx,dy,p) check")
    args = parser.parse_args()

    backend = str(args.backend).lower()

    # Background mesh: structured quads on [-L/2, L/2]^2 (matches step-85).
    nodes, elems, edges, corners = structured_quad(
        float(args.L),
        float(args.L),
        nx=int(args.nx),
        ny=int(args.nx),
        poly_order=1,
        offset=(-0.5 * float(args.L), -0.5 * float(args.L)),
    )
    mesh = PCMesh(nodes, elems, edges, corners, element_type="quad", poly_order=1)

    # deal.II step-85 uses a *discrete* FE_Q(1) level set; mimic with a Q1 surrogate.
    ls_exact = CircleLevelSet(center=(0.0, 0.0), radius=float(args.R))
    ls = PiecewiseLinearLevelSet.from_level_set(mesh, ls_exact)

    # ----------------------------- Scalar (step-85) -----------------------------
    me = MixedElement(mesh, field_specs={"u": int(args.order)})
    dh = DofHandler(me, method="cg")
    dh.classify_from_levelset(ls)
    dh.distribute_dofs_cutfem(ls, domain_side="-", reset=True)

    # Ghost faces in step-85 exclude cut-outside faces (FE_Nothing outside).
    ghost = mesh.edge_bitset("ghost_neg") | mesh.edge_bitset("ghost_both")
    cut = mesh.element_bitset("cut")

    print(
        "[setup] "
        f"nx={args.nx} order={args.order} backend={backend} "
        f"elems={mesh.n_elements} dofs={dh.total_dofs} "
        f"cut_elems={cut.cardinality()} ghost_edges={ghost.cardinality()}"
    )
    if ghost.cardinality() == 0:
        raise RuntimeError("Empty ghost set; try a different --nx/--R combination.")

    u_vec = np.zeros(dh.total_dofs, dtype=float)
    _fill_field(u_vec, dh, "u", lambda x, y: 2.0 - (x * x + y * y))  # exact solution for step-85 data

    du = TrialFunction(field_name="u", name="du", dof_handler=dh)
    v = TestFunction(field_name="u", name="v", dof_handler=dh)
    n = FacetNormal()
    h = CellDiameter()
    gamma = Constant(0.5)  # step-85: ghost_parameter

    dG = dGhost(defined_on=ghost, level_set=ls, metadata={"q": int(args.q_ghost), "derivs": {(1, 0), (0, 1)}})
    dW = dFacetPatch(defined_on=ghost, level_set=ls, metadata={"q": int(args.q_patch)})
    dGamma = dInterface(
        defined_on=cut,
        level_set=ls,
        metadata={"q": int(args.q_interface), "derivs": {(1, 0), (0, 1)}},
    )

    # Facet ghost penalties:
    a_ghost_val = gamma * jump(du) * jump(v) * dG
    a_ghost_grad = gamma * h * jump(grad(du), n) * jump(grad(v), n) * dG

    # Patch/extension stabilization:
    a_patch_val = gamma * jump(du) * jump(v) * dW

    # Nitsche interface term (deal.II step-85):
    beta = Constant(5.0 * float(int(args.order) + 1) * float(int(args.order)))
    a_ifc = (-dot(grad(du), n) * v - dot(grad(v), n) * du + (beta / h) * du * v) * dGamma

    E_ghost_val = assemble_energy(a_ghost_val, dh, u_vec, backend=backend)
    E_ghost_grad = assemble_energy(a_ghost_grad, dh, u_vec, backend=backend)
    E_patch_val = assemble_energy(a_patch_val, dh, u_vec, backend=backend)
    E_ifc = assemble_energy(a_ifc, dh, u_vec, backend=backend)

    print("\nScalar (step-85-style) energies:  uᵀ K u")
    print(f"  dGhost      jump(u)         : {E_ghost_val:+.12e}  (CG value jump → ~0)")
    print(f"  dGhost      jump(grad(u),n) : {E_ghost_grad:+.12e}  (deal.II step-85 ghost)")
    print(f"  dFacetPatch jump(u)         : {E_patch_val:+.12e}  (patch/extension, nonzero for CG)")
    print(f"  dInterface  Nitsche         : {E_ifc:+.12e}")

    if not bool(args.no_manual_check):
        E_manual = manual_ghost_energy_q1(
            mesh=mesh,
            dh=dh,
            me=me,
            u_vec=u_vec,
            ghost_edges=ghost,
            gamma=0.5,
            q=int(args.q_ghost),
            h_mode=str(args.manual_h),
        )
        print("\nManual check (deal.II-style ghost energy):")
        _print_abs_rel(f"manual (h_mode={args.manual_h})", E_manual, E_ghost_grad)

    # ----------------------------- Mixed fields -----------------------------
    if args.mixed:
        print("\nMixed-field ghost/patch energies (ux,uy,dx,dy,p):  uᵀ K u")
        fields = {"ux": 2, "uy": 2, "dx": 2, "dy": 2, "p": 1}
        me_m = MixedElement(mesh, field_specs=fields)
        dh_m = DofHandler(me_m, method="cg")
        dh_m.classify_from_levelset(ls)
        dh_m.distribute_dofs_cutfem(ls, domain_side="-", reset=True)

        u_m = np.zeros(dh_m.total_dofs, dtype=float)
        # Use non-representable (for Q2/Q1) quartic probes so ghost/patch energies are non-zero.
        _fill_field(u_m, dh_m, "ux", lambda x, y: x**4 + 0.5 * y + 0.1 * x * y)
        _fill_field(u_m, dh_m, "uy", lambda x, y: y**4 - 0.3 * x + 0.05 * x * y)
        _fill_field(u_m, dh_m, "dx", lambda x, y: 0.1 * x**4 - 0.2 * x * y + 0.05 * y**4)
        _fill_field(u_m, dh_m, "dy", lambda x, y: -0.07 * y**4 + 0.3 * x * y + 0.02 * x**4)
        _fill_field(u_m, dh_m, "p", lambda x, y: x - y + 0.2 * x * x + 0.1 * y**4)

        Vv = FunctionSpace("Vv", ["ux", "uy"])
        Vd = FunctionSpace("Vd", ["dx", "dy"])
        uv = VectorTrialFunction(Vv, dh_m)
        vv = VectorTestFunction(Vv, dh_m)
        ud = VectorTrialFunction(Vd, dh_m)
        vd = VectorTestFunction(Vd, dh_m)
        dp = TrialFunction(field_name="p", name="dp", dof_handler=dh_m)
        q = TestFunction(field_name="p", name="q", dof_handler=dh_m)

        n = FacetNormal()
        h = CellDiameter()
        gamma = Constant(0.5)

        dG_m = dGhost(defined_on=ghost, level_set=ls, metadata={"q": int(args.q_ghost), "derivs": {(1, 0), (0, 1)}})
        dW_m = dFacetPatch(defined_on=ghost, level_set=ls, metadata={"q": int(args.q_patch)})

        a_vel_ghost = gamma * h * (
            jump(grad(uv[0]), n) * jump(grad(vv[0]), n) + jump(grad(uv[1]), n) * jump(grad(vv[1]), n)
        ) * dG_m
        a_vel_patch = gamma * (jump(uv[0]) * jump(vv[0]) + jump(uv[1]) * jump(vv[1])) * dW_m

        a_disp_ghost = gamma * h * (
            jump(grad(ud[0]), n) * jump(grad(vd[0]), n) + jump(grad(ud[1]), n) * jump(grad(vd[1]), n)
        ) * dG_m
        a_disp_patch = gamma * (jump(ud[0]) * jump(vd[0]) + jump(ud[1]) * jump(vd[1])) * dW_m

        a_p_ghost = gamma * h * jump(grad(dp), n) * jump(grad(q), n) * dG_m
        a_p_patch = gamma * jump(dp) * jump(q) * dW_m

        E_vel_ghost = assemble_energy(a_vel_ghost, dh_m, u_m, backend=backend)
        E_vel_patch = assemble_energy(a_vel_patch, dh_m, u_m, backend=backend)
        E_disp_ghost = assemble_energy(a_disp_ghost, dh_m, u_m, backend=backend)
        E_disp_patch = assemble_energy(a_disp_patch, dh_m, u_m, backend=backend)
        E_p_ghost = assemble_energy(a_p_ghost, dh_m, u_m, backend=backend)
        E_p_patch = assemble_energy(a_p_patch, dh_m, u_m, backend=backend)

        print(f"  velocity     : dGhost grad-jump={E_vel_ghost:+.12e}  dFacetPatch val-jump={E_vel_patch:+.12e}")
        print(f"  displacement : dGhost grad-jump={E_disp_ghost:+.12e}  dFacetPatch val-jump={E_disp_patch:+.12e}")
        print(f"  pressure     : dGhost grad-jump={E_p_ghost:+.12e}  dFacetPatch val-jump={E_p_patch:+.12e}")


if __name__ == "__main__":
    main()
