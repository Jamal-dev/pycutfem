#!/usr/bin/env python3
"""
Finite-difference consistency check for the semi-smooth (active-set) contact term.

This script assembles:
  - the contact residual vector R(u)
  - the semi-smooth Newton Jacobian A(u)
and verifies A(u)·δ ≈ (R(u+ε δ) - R(u))/ε.

It is intentionally small and backend-agnostic so we can regression-test
`PositivePart`/`Heaviside` and their usage pattern (residual vs Jacobian)
across pycutfem backends.
"""

from __future__ import annotations

import numpy as np

from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.levelset import CircleLevelSet
from pycutfem.fem.mixedelement import MixedElement

from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.expressions import (
    Constant,
    TrialFunction,
    TestFunction,
    VectorTrialFunction,
    VectorTestFunction,
    Function,
    VectorFunction,
    FacetNormal,
    CellDiameter,
)
from pycutfem.ufl.measures import dInterface
from pycutfem.solvers.nonlinear_solver import NewtonSolver, NewtonParameters
from examples.utils.fsi.contact import (
    RelaxedWallContact,
    sigma_f_newtonian,
    dsigma_f_newtonian,
    sigma_s_stvk,
    dsigma_s_stvk,
)
from pycutfem.ufl.analytic import Analytic, y


def _assemble_contact_only(*, backend: str, wall_eps: float):
    # --- geometry / mesh -------------------------------------------------
    Lx, Ly = 1.0, 1.0
    nx, ny = 8, 8
    poly_order = 2
    nodes, elems, _, corners = structured_quad(Lx, Ly, nx=nx, ny=ny, poly_order=poly_order)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )

    # --- level set: circle inside the box ------------------------------
    level_set = CircleLevelSet(center=(0.5, 0.6), radius=0.2)
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)
    cut_e = mesh.element_bitset("cut")

    # --- FE / dofs ------------------------------------------------------
    deg_v = 2
    deg_p = 1
    deg_s = 1
    fields = {
        "vf_x": deg_v,
        "vf_y": deg_v,
        "pf": deg_p,
        "vs_x": deg_s,
        "vs_y": deg_s,
        "u_x": deg_s,
        "u_y": deg_s,
    }
    me = MixedElement(mesh, field_specs=fields)
    dh = DofHandler(me, method="cg")

    # spaces
    Vf = FunctionSpace("Vf", ["vf_x", "vf_y"])
    Pf = FunctionSpace("Pf", ["pf"])
    Vs = FunctionSpace("Vs", ["vs_x", "vs_y"])
    Us = FunctionSpace("Us", ["u_x", "u_y"])

    # trials (directional increments)
    dvf = VectorTrialFunction(Vf, dof_handler=dh)
    dpf = TrialFunction(name="trial_pf", field_name="pf", dof_handler=dh)
    dvs = VectorTrialFunction(Vs, dof_handler=dh)
    du = VectorTrialFunction(Us, dof_handler=dh)

    # tests
    tvs = VectorTestFunction(Vs, dof_handler=dh)

    # state (current / previous)
    vf = VectorFunction("vf", ["vf_x", "vf_y"], dh)
    pf = Function("pf", "pf", dh)
    vs = VectorFunction("vs", ["vs_x", "vs_y"], dh)
    u = VectorFunction("u", ["u_x", "u_y"], dh)
    u_prev = VectorFunction("u_prev", ["u_x", "u_y"], dh)

    # deterministic pseudo-random state (avoid P ~ 0 by scaling)
    rng = np.random.default_rng(0)
    for f in (vf, pf, vs, u):
        f.nodal_values[:] = 1e-2 * rng.standard_normal(f.nodal_values.shape)
    u_prev.nodal_values[:] = 0.0

    # --- contact parameters --------------------------------------------
    k = Constant(1.0e-3)
    rho_f = Constant(1.0)
    nu_f = Constant(1.0e-2)
    mu_s = Constant(2.0)
    lambda_s = Constant(3.0)

    gamma_N = Constant(1.0e2)
    gamma_C = Constant(5.0e1)
    eps_wall = Constant(float(wall_eps))

    n_s = FacetNormal()
    n_f = Constant(-1.0) * n_s
    h = CellDiameter()
    penalty_nitsche = rho_f * nu_f * gamma_N / h

    gap_eps = Analytic(y) - eps_wall  # bottom wall at y=0 → g0=y

    contact = RelaxedWallContact(
        gamma_C=gamma_C,
        gap_eps=gap_eps,
        n_s=n_s,
        n_f=n_f,
        penalty_nitsche=penalty_nitsche,
    )

    sig_f = sigma_f_newtonian(vf, pf, rho_f=rho_f, nu_f=nu_f)
    sig_s = sigma_s_stvk(u, mu_s=mu_s, lambda_s=lambda_s)
    P = contact.P_gammaC(
        u=u,
        u_prev=u_prev,
        v_f=vf,
        p_f=pf,
        v_s=vs,
        sigma_s=sig_s,
        sigma_f=sig_f,
    )

    dP = contact.dP_gammaC(
        du=du,
        dv_f=dvf,
        dp_f=dpf,
        dv_s=dvs,
        dsigma_s=dsigma_s_stvk(u, du, mu_s=mu_s, lambda_s=lambda_s),
        dsigma_f=dsigma_f_newtonian(dvf, dpf, rho_f=rho_f, nu_f=nu_f),
    )

    dΓ = dInterface(
        defined_on=cut_e,
        level_set=level_set,
        metadata={"q": 6, "derivs": {(0, 0), (1, 0), (0, 1)}},
    )

    residual_form = contact.residual_term(P=P, k=k, test_v_s=tvs) * dΓ
    jacobian_form = contact.jacobian_term(P=P, dP=dP, k=k, test_v_s=tvs) * dΓ

    solver = NewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(newton_tol=1.0e-14, line_search=False),
        backend=backend,
        quad_order=6,
    )

    funcs = [vf, pf, vs, u]
    coeffs = {f.name: f for f in funcs}
    coeffs[u_prev.name] = u_prev

    A, R = solver._assemble_system(coeffs, need_matrix=True)  # pylint: disable=protected-access

    # FD check ---------------------------------------------------------
    ndof = dh.total_dofs
    delta = rng.standard_normal(ndof)
    delta *= 1.0e-3 / (np.linalg.norm(delta, np.inf) + 1.0e-16)

    eps_fd = 1.0e-7
    # R(u + eps*delta)
    snap = [f.nodal_values.copy() for f in funcs]
    try:
        dh.add_to_functions(eps_fd * delta, funcs)
        A1, R1 = solver._assemble_system(coeffs, need_matrix=False)  # type: ignore[assignment]
        assert A1 is None
    finally:
        for f, buf in zip(funcs, snap):
            f.nodal_values[:] = buf

    dR_fd = (R1 - R) / eps_fd
    dR_lin = A @ delta

    err = np.linalg.norm(dR_fd - dR_lin, np.inf)
    ref = max(1.0, np.linalg.norm(dR_fd, np.inf), np.linalg.norm(dR_lin, np.inf))
    rel = err / ref
    return float(err), float(rel), int(ndof)


def main():
    cases = (
        ("inactive (P<0)", 1.0e-2),
        ("active (P>0)", 1.0),
    )
    for label, wall_eps in cases:
        print(f"\n--- {label}: wall_eps={wall_eps:g} ---")
        for backend in ("python", "jit", "cpp"):
            try:
                err, rel, ndof = _assemble_contact_only(backend=backend, wall_eps=wall_eps)
            except Exception as exc:  # pragma: no cover - debug script
                print(f"[{backend:6}] FAILED: {exc}")
                continue
            print(f"[{backend:6}] ndof={ndof:5d}  |FD - A·d|_inf={err:.3e}  rel={rel:.3e}")


if __name__ == "__main__":
    main()
