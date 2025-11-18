#!/usr/bin/env python3
"""Compare Stokes interface assembly between python and JIT backends.

This script builds the `examples.stokes_cut` problem (with deformation) and
assembles each integral separately with both backends.  It reports per-integral
matrix/vector differences and inspects representative geometric factors
(`Ji`, `det_t`) on a selected cut element.

Run directly to obtain a plain-text report:

    python examples/debug_tools/stokes_backend_diagnostics.py \
        --with-deformation --sample-eid 0 --geom-side +

Use `--detail` for verbose matrix norms, `--save-json` to dump the raw data.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse
import sympy as sp

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.core.levelset import CircleLevelSet, LevelSetMeshAdaptation
from pycutfem.core.geometry import hansbo_cut_ratio
from pycutfem.core.sideconvention import SIDE
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.fem import transform
from pycutfem.integration.cut_integration import CutIntegration
try:
    from pycutfem.integration.quadrature import isoparam_interface_line_quadrature_batch as iso_ifc_rule
except ImportError:
    iso_ifc_rule = None
from pycutfem.integration.quadrature import curved_line_quadrature_batch
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import (
    TrialFunction,
    TestFunction,
    VectorTrialFunction,
    VectorTestFunction,
    Constant,
    grad,
    inner,
    dot,
    div,
    Pos,
    Neg,
    FacetNormal,
    CellDiameter,
    ElementWiseConstant,
    restrict,
)
from pycutfem.ufl.forms import Equation, Form, assemble_form
from pycutfem.ufl.helpers import HelpersFieldAware as _hfa
from pycutfem.ufl.measures import dx, dInterface, dGhost
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


@dataclass
class ProblemData:
    equation: Equation
    dh: DofHandler
    level_set: CircleLevelSet
    deformation: object | None
    mesh: Mesh
    qvol: int
    geom_order: int
    lhs_labels: List[str]
    rhs_labels: List[str]


def exact_solution(mu, R, gammaf):
    sx, sy = sp.symbols('x y', real=True)
    r2 = sx * sx + sy * sy

    aneg = 1.0 / mu[0].value
    apos_base = 1.0 / mu[1].value
    apos = apos_base + (aneg - apos_base) * sp.exp(r2 - R * R)

    u_neg_x = -aneg * sy * sp.exp(-r2)
    u_neg_y = aneg * sx * sp.exp(-r2)
    u_pos_x = -apos * sy * sp.exp(-r2)
    u_pos_y = apos * sx * sp.exp(-r2)

    p_neg_sym = sx ** 3
    p_pos_sym = sx ** 3 - gammaf

    def eps_mat(ux, uy):
        du = sp.Matrix([[sp.diff(ux, sx), sp.diff(ux, sy)],
                        [sp.diff(uy, sx), sp.diff(uy, sy)]])
        return 0.5 * (du + du.T)

    I2 = sp.eye(2)
    eps_neg = eps_mat(u_neg_x, u_neg_y)
    eps_pos = eps_mat(u_pos_x, u_pos_y)

    sigma_neg = -2 * mu[0].value * eps_neg + p_neg_sym * I2
    sigma_pos = -2 * mu[1].value * eps_pos + p_pos_sym * I2

    def div_sigma(sig):
        return (sp.diff(sig[0, 0], sx) + sp.diff(sig[0, 1], sy),
                sp.diff(sig[1, 0], sx) + sp.diff(sig[1, 1], sy))

    g_neg_x_sym, g_neg_y_sym = div_sigma(sigma_neg)
    g_pos_x_sym, g_pos_y_sym = div_sigma(sigma_pos)

    u_neg_x_fun = sp.lambdify((sx, sy), u_neg_x, 'numpy')
    u_neg_y_fun = sp.lambdify((sx, sy), u_neg_y, 'numpy')
    u_pos_x_fun = sp.lambdify((sx, sy), u_pos_x, 'numpy')
    u_pos_y_fun = sp.lambdify((sx, sy), u_pos_y, 'numpy')

    p_neg_fun = sp.lambdify((sx, sy), p_neg_sym, 'numpy')
    p_pos_fun = sp.lambdify((sx, sy), p_pos_sym, 'numpy')

    dux_dx_neg = sp.lambdify((sx, sy), sp.diff(u_neg_x, sx), 'numpy')
    dux_dy_neg = sp.lambdify((sx, sy), sp.diff(u_neg_x, sy), 'numpy')
    duy_dx_neg = sp.lambdify((sx, sy), sp.diff(u_neg_y, sx), 'numpy')
    duy_dy_neg = sp.lambdify((sx, sy), sp.diff(u_neg_y, sy), 'numpy')

    dux_dx_pos = sp.lambdify((sx, sy), sp.diff(u_pos_x, sx), 'numpy')
    dux_dy_pos = sp.lambdify((sx, sy), sp.diff(u_pos_x, sy), 'numpy')
    duy_dx_pos = sp.lambdify((sx, sy), sp.diff(u_pos_y, sx), 'numpy')
    duy_dy_pos = sp.lambdify((sx, sy), sp.diff(u_pos_y, sy), 'numpy')

    g_neg_x_fun = sp.lambdify((sx, sy), g_neg_x_sym, 'numpy')
    g_neg_y_fun = sp.lambdify((sx, sy), g_neg_y_sym, 'numpy')
    g_pos_x_fun = sp.lambdify((sx, sy), g_pos_x_sym, 'numpy')
    g_pos_y_fun = sp.lambdify((sx, sy), g_pos_y_sym, 'numpy')

    def vec2_callable(fx, fy):
        def f(x, y):
            ax = np.asarray(fx(x, y), dtype=float)
            ay = np.asarray(fy(x, y), dtype=float)
            return np.stack([ax, ay], axis=-1)
        return f

    def grad2x2_callable(fxx, fxy, fyx, fyy):
        def g(x, y):
            a = np.asarray(fxx(x, y), dtype=float)
            b = np.asarray(fxy(x, y), dtype=float)
            c = np.asarray(fyx(x, y), dtype=float)
            d = np.asarray(fyy(x, y), dtype=float)
            return np.stack([np.stack([a, b], axis=-1),
                             np.stack([c, d], axis=-1)], axis=-2)
        return g

    vel_exact_neg_xy = vec2_callable(u_neg_x_fun, u_neg_y_fun)
    vel_exact_pos_xy = vec2_callable(u_pos_x_fun, u_pos_y_fun)

    grad_vel_neg_xy = grad2x2_callable(dux_dx_neg, dux_dy_neg, duy_dx_neg, duy_dy_neg)
    grad_vel_pos_xy = grad2x2_callable(dux_dx_pos, dux_dy_pos, duy_dx_pos, duy_dy_pos)

    def p_exact_neg_xy(x, y):
        return np.asarray(p_neg_fun(x, y), dtype=float)

    def p_exact_pos_xy(x, y):
        return np.asarray(p_pos_fun(x, y), dtype=float)

    g_neg_xy = vec2_callable(g_neg_x_fun, g_neg_y_fun)
    g_pos_xy = vec2_callable(g_pos_x_fun, g_pos_y_fun)

    return (vel_exact_neg_xy, vel_exact_pos_xy,
            g_neg_xy, g_pos_xy,
            grad_vel_neg_xy, grad_vel_pos_xy,
            p_exact_neg_xy, p_exact_pos_xy)


def build_stokes_problem(with_deformation: bool) -> ProblemData:
    poly_order = 2
    geom_order = 2 if with_deformation else 1
    qvol = (2 * poly_order + 4) if with_deformation else (2 * poly_order + 2)

    maxh = 0.125
    L, H = 2.0, 2.0
    mesh_size = int(L / maxh)

    nodes, elems, _, corners = structured_quad(
        L,
        H,
        nx=mesh_size,
        ny=mesh_size,
        poly_order=geom_order,
        offset=[-L / 2.0, -H / 2.0],
    )
    mesh = Mesh(
        nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        poly_order=geom_order,
        element_type="quad",
    )

    mu = [Constant(1.0), Constant(10.0)]
    R = 2.0 / 3.0
    gammaf = 0.5
    level_set = CircleLevelSet(center=(0.0, 0.0), radius=R)

    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)

    me = MixedElement(
        mesh,
        field_specs={
            "u_pos_x": poly_order,
            "u_pos_y": poly_order,
            "p_pos_": poly_order - 1,
            "u_neg_x": poly_order,
            "u_neg_y": poly_order,
            "p_neg_": poly_order - 1,
            "lm": ":number:",
        },
    )
    dh = DofHandler(me, method="cg")

    boundary_tags = {
        "boundary": lambda x, y: np.isclose(x, -L / 2.0)
        | np.isclose(x, L / 2.0)
        | np.isclose(y, -H / 2.0)
        | np.isclose(y, H / 2.0)
    }
    dh.tag_dofs_by_locator_map(boundary_tags, fields=["u_pos_x", "u_pos_y", "u_neg_x", "u_neg_y"])

    Vspace_pos = FunctionSpace("vel_positive", ["u_pos_x", "u_pos_y"])
    Vspace_neg = FunctionSpace("vel_negative", ["u_neg_x", "u_neg_y"])
    vel_trial_pos = VectorTrialFunction(space=Vspace_pos, dof_handler=dh, side="+")
    vel_test_pos = VectorTestFunction(space=Vspace_pos, dof_handler=dh, side="+")
    vel_trial_neg = VectorTrialFunction(space=Vspace_neg, dof_handler=dh, side="-")
    vel_test_neg = VectorTestFunction(space=Vspace_neg, dof_handler=dh, side="-")
    p_trial_pos = TrialFunction("p_pos_", dh, name="pressure_pos_trial", side="+")
    q_test_pos = TestFunction("p_pos_", dh, name="pressure_pos_test", side="+")
    p_trial_neg = TrialFunction("p_neg_", dh, name="pressure_neg_trial", side="-")
    q_test_neg = TestFunction("p_neg_", dh, name="pressure_neg_test", side="-")
    nL = TrialFunction("lm")
    mL = TestFunction("lm")

    inside_e = mesh.element_bitset("inside")
    outside_e = mesh.element_bitset("outside")
    cut_e = mesh.element_bitset("cut")
    ghost_pos = mesh.edge_bitset("ghost_pos") | mesh.edge_bitset("ghost_both") | mesh.edge_bitset("interface")
    ghost_neg = mesh.edge_bitset("ghost_neg") | mesh.edge_bitset("ghost_both") | mesh.edge_bitset("interface")

    has_inside = inside_e | cut_e
    has_outside = outside_e | cut_e

    deformation = None
    if with_deformation:
        adapter = LevelSetMeshAdaptation(mesh, order=max(2, poly_order), threshold=10.5)
        deformation = adapter.calc_deformation(level_set, q_vol=qvol)
        level_set = adapter.lset_p1
        mesh.classify_elements(level_set)
        mesh.classify_edges(level_set)
        mesh.build_interface_segments(level_set)

    dx_pos = dx(
        defined_on=has_outside,
        level_set=level_set,
        metadata={"side": "+", "q": qvol},
        deformation=deformation,
    )
    dx_neg = dx(
        defined_on=has_inside,
        level_set=level_set,
        metadata={"side": "-", "q": qvol},
        deformation=deformation,
    )
    dGamma = dInterface(
        defined_on=cut_e,
        level_set=level_set,
        metadata={"q": qvol + 2, "derivs": {(1, 0), (0, 1)}},
        deformation=deformation,
    )
    dG_pos = dGhost(
        defined_on=ghost_pos,
        level_set=level_set,
        metadata={"q": qvol + 2, "derivs": {(1, 0), (0, 1)}},
        deformation=deformation,
    )
    dG_neg = dGhost(
        defined_on=ghost_neg,
        level_set=level_set,
        metadata={"q": qvol + 2, "derivs": {(1, 0), (0, 1)}},
        deformation=deformation,
    )

    h = CellDiameter()
    n = FacetNormal()

    theta_pos_vals = hansbo_cut_ratio(mesh, level_set, side="+")
    theta_neg_vals = 1.0 - theta_pos_vals
    kappa_pos = Pos(ElementWiseConstant(theta_pos_vals))
    kappa_neg = Neg(ElementWiseConstant(theta_neg_vals))

    lambda_nitsche = Constant(0.5 * (mu[0].value + mu[1].value) * 20 * poly_order**2)
    gamma_stab_v = Constant(0.05)
    gamma_stab_p = Constant(0.05)

    (
        vel_exact_neg_xy,
        vel_exact_pos_xy,
        g_neg_xy,
        g_pos_xy,
        grad_vel_neg_xy,
        grad_vel_pos_xy,
        p_exact_neg_xy,
        p_exact_pos_xy,
    ) = exact_solution(mu, R, gammaf)
    g_neg = Analytic(g_neg_xy)
    g_pos = Analytic(g_pos_xy)

    def epsilon(u):
        return 0.5 * (grad(u) + grad(u).T)

    def traction(mu_val, u_vec, p_scal, normal):
        return -2 * mu_val * dot(epsilon(u_vec), normal) + p_scal * normal

    q_pos_R = restrict(q_test_pos, has_outside)
    p_pos_R = restrict(p_trial_pos, has_outside)
    q_neg_R = restrict(q_test_neg, has_inside)
    p_neg_R = restrict(p_trial_neg, has_inside)

    lhs_terms = []
    lhs_labels = []

    term_pos_vol = (
        2 * mu[1] * inner(epsilon(vel_trial_pos), epsilon(vel_test_pos))
        - div(vel_trial_pos) * q_pos_R
        - div(vel_test_pos) * p_pos_R
    ) * dx_pos
    lhs_terms.append(term_pos_vol)
    lhs_labels.append("vol_pos")

    term_neg_vol = (
        2 * mu[0] * inner(epsilon(vel_trial_neg), epsilon(vel_test_neg))
        - div(vel_trial_neg) * q_neg_R
        - div(vel_test_neg) * p_neg_R
    ) * dx_neg
    lhs_terms.append(term_neg_vol)
    lhs_labels.append("vol_neg")

    term_lm = (nL * Neg(q_neg_R) + mL * Neg(p_neg_R)) * dx_neg
    lhs_terms.append(term_lm)
    lhs_labels.append("lagrange_neg")

    traction_pos_n_trial = traction(mu[1], Pos(vel_trial_pos), Pos(p_trial_pos), n)
    traction_neg_n_trial = traction(mu[0], Neg(vel_trial_neg), Neg(p_trial_neg), n)
    avg_flux_trial = kappa_pos * traction_pos_n_trial + kappa_neg * traction_neg_n_trial

    traction_pos_n_test = traction(mu[1], Pos(vel_test_pos), Pos(q_test_pos), n)
    traction_neg_n_test = traction(mu[0], Neg(vel_test_neg), Neg(q_test_neg), n)
    avg_flux_test = kappa_pos * traction_pos_n_test + kappa_neg * traction_neg_n_test

    def jump_ng(u_pos, u_neg):
        return Neg(u_neg) - Pos(u_pos)

    jump_vel_trial = jump_ng(vel_trial_pos, vel_trial_neg)
    jump_vel_test = jump_ng(vel_test_pos, vel_test_neg)

    term_flux_trial = dot(avg_flux_trial, jump_vel_test) * dGamma
    lhs_terms.append(term_flux_trial)
    lhs_labels.append("ifc_flux_trial_jump")

    term_flux_test = dot(avg_flux_test, jump_vel_trial) * dGamma
    lhs_terms.append(term_flux_test)
    lhs_labels.append("ifc_flux_test_jump")

    term_penalty = ((lambda_nitsche / 0.125) * dot(jump_vel_trial, jump_vel_test)) * dGamma
    lhs_terms.append(term_penalty)
    lhs_labels.append("ifc_penalty")

    avg_inv_test = kappa_neg * Pos(vel_test_pos) + kappa_pos * Neg(vel_test_neg)

    rhs_terms = []
    rhs_labels = []

    term_body_pos = dot(g_pos, vel_test_pos) * dx_pos
    rhs_terms.append(term_body_pos)
    rhs_labels.append("rhs_body_pos")

    term_body_neg = dot(g_neg, vel_test_neg) * dx_neg
    rhs_terms.append(term_body_neg)
    rhs_labels.append("rhs_body_neg")

    term_surface = -gammaf * dot(avg_inv_test, n) * dGamma
    rhs_terms.append(term_surface)
    rhs_labels.append("rhs_surface_tension")

    a = Form(lhs_terms)
    f = Form(rhs_terms)
    equation = Equation(a, f)

    return ProblemData(
        equation=equation,
        dh=dh,
        level_set=level_set,
        deformation=deformation,
        mesh=mesh,
        qvol=qvol,
        geom_order=geom_order,
        lhs_labels=lhs_labels,
        rhs_labels=rhs_labels,
    )


def assemble_integral_contributions(problem: ProblemData, backend: str) -> Dict[str, List[Dict[str, object]]]:
    dh = problem.dh
    eq = problem.equation
    results = {"lhs": [], "rhs": []}

    def _assemble_single(form: Form | None, labels: List[str], rhs: bool) -> List[Dict[str, object]]:
        if form is None:
            return []
        out: List[Dict[str, object]] = []
        for idx, integral in enumerate(form.integrals):
            single = Form([integral])
            lhs = single if not rhs else None
            rhs_form = single if rhs else None
            local_eq = Equation(lhs, rhs_form)
            K, F = assemble_form(local_eq, dof_handler=dh, bcs=[], backend=backend)
            label = labels[idx] if idx < len(labels) else f"term_{idx}"
            common = {
                "index": idx,
                "label": label,
                "repr": repr(integral.integrand),
                "measure": integral.measure.domain_type,
            }
            if rhs:
                out.append({**common, "vector": F.copy()})
            else:
                out.append({**common, "matrix": K.copy()})
        return out

    results["lhs"] = _assemble_single(problem.equation.a, problem.lhs_labels, rhs=False)
    results["rhs"] = _assemble_single(problem.equation.L, problem.rhs_labels, rhs=True)
    return results


def _norm_diff(A: scipy.sparse.spmatrix, B: scipy.sparse.spmatrix) -> Tuple[float, float]:
    AX = A.toarray()
    BX = B.toarray()
    diff = AX - BX
    fro = float(np.linalg.norm(diff))
    max_abs = float(np.max(np.abs(diff)))
    denom = max(float(np.linalg.norm(AX)), 1e-14)
    rel = fro / denom
    return max_abs, rel


def _vec_diff(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    diff = a - b
    max_abs = float(np.max(np.abs(diff)))
    denom = max(float(np.linalg.norm(a)), 1e-14)
    rel = float(np.linalg.norm(diff)) / denom
    return max_abs, rel


def compare_contributions(py_data, jit_data, detail: bool = False) -> Dict[str, list]:
    report = {"lhs": [], "rhs": []}

    for part in ("lhs", "rhs"):
        py_terms = py_data.get(part, [])
        jit_terms = jit_data.get(part, [])
        if len(py_terms) != len(jit_terms):
            raise RuntimeError(f"Mismatch in number of {part} integrals ({len(py_terms)} vs {len(jit_terms)})")
        for py_term, jit_term in zip(py_terms, jit_terms):
            assert py_term["index"] == jit_term["index"]
            if part == "lhs":
                max_abs, rel = _norm_diff(py_term["matrix"], jit_term["matrix"])
            else:
                max_abs, rel = _vec_diff(py_term["vector"], jit_term["vector"])
            entry = {
                "index": py_term["index"],
                "measure": py_term["measure"],
                "label": py_term.get("label", f"term_{py_term['index']}"),
                "integrand": py_term["repr"],
                "max_abs": max_abs,
                "rel_fro": rel,
            }
            report[part].append(entry)
            if detail:
                print(
                    f"[{part.upper()}] term {py_term['index']:02d} ({py_term['measure']}, {entry['label']})\n"
                    f"    max|Δ| = {max_abs:.3e}, relative Frobenius = {rel:.3e}\n"
                    f"    integrand: {py_term['repr']}"
                )
    return report

def _straight_cut_quadrature(mesh: Mesh, eid: int, level_set, qdeg: int, side: str) -> Tuple[np.ndarray, np.ndarray]:
    order_y = max(2, qdeg // 2)
    order_x = max(2, qdeg // 2)
    qpref, qwref = CutIntegration.straight_cut_rule_quad_ref(
        mesh,
        int(eid),
        level_set,
        side=side,
        order_y=order_y,
        order_x=order_x,
        tol=SIDE.tol,
    )
    return np.asarray(qpref, float), np.asarray(qwref, float)


def collect_python_geom(mesh: Mesh, deformation, level_set, eid: int, qdeg: int, side: str) -> Dict[str, np.ndarray]:
    ref_geom = transform.get_reference(mesh.element_type, mesh.poly_order)
    qpref, qwref = _straight_cut_quadrature(mesh, eid, level_set, qdeg, side)
    Ji = []
    det = []
    pts = []
    for (xi, eta), w in zip(qpref, qwref):
        Jg = transform.jacobian(mesh, int(eid), (float(xi), float(eta)))
        xg = transform.x_mapping(mesh, int(eid), (float(xi), float(eta)))
        if deformation is None:
            Ji_val = np.linalg.inv(Jg)
            Ji.append(Ji_val)
            det.append(abs(float(1.0 / np.linalg.det(Ji_val))))
            pts.append(np.asarray(xg, float))
            continue
        conn = np.asarray(mesh.elements_connectivity[int(eid)], dtype=int)
        dN = np.asarray(ref_geom.grad(float(xi), float(eta)), float)
        Uloc = np.asarray(deformation.node_displacements[conn], float)
        Jg_inv = np.linalg.inv(Jg)
        G_phys = (Uloc.T @ dN) @ Jg_inv
        F = np.eye(mesh.spatial_dim) + G_phys
        Finv = np.linalg.inv(F)
        Ji_val = Jg_inv @ Finv
        Ji.append(Ji_val)
        det.append(abs(float(1.0 / np.linalg.det(Ji_val))))
        disp = deformation.displacement_ref(int(eid), (float(xi), float(eta)))
        pts.append(np.asarray(xg, float) + np.asarray(disp, float))
    return {
        "xi_eta": qpref,
        "weights": qwref,
        "Ji": np.asarray(Ji, float),
        "det": np.asarray(det, float),
        "points": np.asarray(pts, float),
    }


def collect_jit_geom(dh: DofHandler, level_set, deformation, eid: int, qdeg: int, side: str) -> Dict[str, np.ndarray]:
    derivs = {(0, 0)}
    geo = dh.precompute_cut_volume_factors(
        np.array([eid], dtype=np.int32),
        qdeg,
        derivs,
        level_set,
        side=side,
        deformation=deformation,
        reuse=False,
    )
    if geo["qp_phys"].size == 0:
        return {"qp_phys": np.zeros((0, 2)), "Ji": np.zeros((0, 2, 2)), "det": np.zeros((0,))}
    pts = geo["qp_phys"][0]
    Ji = geo["J_inv"][0]
    det = np.array([abs(float(1.0 / np.linalg.det(M))) for M in Ji], dtype=float)
    return {"qp_phys": pts, "Ji": Ji, "det": det}




def interface_geometry_report(problem: ProblemData, sample_eid: int, qdeg: int, detail: bool = False) -> Dict[str, np.ndarray]:
    mesh = problem.mesh
    level_set = problem.level_set
    deformation = problem.deformation
    if sample_eid < 0 or sample_eid >= mesh.n_elements:
        raise ValueError(f"sample_eid {sample_eid} outside mesh range (0..{mesh.n_elements - 1})")

    elem = mesh.elements_list[int(sample_eid)]
    if getattr(elem, "tag", None) != "cut":
        raise ValueError(f"Element {sample_eid} is not marked as cut; cannot extract interface data.")

    P0, P1 = elem.interface_pts
    P0_arr = np.asarray([P0], dtype=float)
    P1_arr = np.asarray([P1], dtype=float)

    if iso_ifc_rule is not None:
        qb, wb, tb, rb = iso_ifc_rule(
            level_set,
            P0_arr,
            P1_arr,
            p=max(2, mesh.poly_order) if deformation is not None else mesh.poly_order,
            order=int(qdeg),
            project_steps=3,
            tol=SIDE.tol,
            mesh=mesh,
            eids=np.asarray([sample_eid], dtype=int),
            return_tangent=True,
            return_qref=True,
        )
        qpts = np.asarray(qb[0], dtype=float)
        qw = np.asarray(wb[0], dtype=float)
        tangents = np.asarray(tb[0], dtype=float) if tb is not None else None
        qref = np.asarray(rb[0], dtype=float) if rb is not None else None
    else:
        nseg = max(3, mesh.poly_order + qdeg // 2)
        qpts_batch, qw_batch = curved_line_quadrature_batch(
            level_set, P0_arr, P1_arr, order=int(qdeg), nseg=nseg, project_steps=3, tol=SIDE.tol
        )
        qpts = np.asarray(qpts_batch[0], dtype=float)
        qw = np.asarray(qw_batch[0], dtype=float)
        tangents = None
        qref = None

    ref_geom = transform.get_reference(mesh.element_type, mesh.poly_order)
    Ji_vals = []
    w_eff = []
    tau_norms = []
    for iq, (x_phys, w) in enumerate(zip(qpts, qw)):
        if qref is not None:
            xi = float(qref[iq, 0])
            eta = float(qref[iq, 1])
        else:
            xi, eta = transform.inverse_mapping(mesh, int(sample_eid), x_phys)
        Jg = transform.jacobian(mesh, int(sample_eid), (xi, eta))
        if deformation is None:
            Ji = np.linalg.inv(Jg)
            w_here = float(w)
        else:
            conn = np.asarray(mesh.elements_connectivity[int(sample_eid)], dtype=int)
            dN = np.asarray(ref_geom.grad(float(xi), float(eta)), float)
            Uloc = np.asarray(deformation.node_displacements[conn], float)
            Jg_inv = np.linalg.inv(Jg)
            G_phys = (Uloc.T @ dN) @ Jg_inv
            F = np.eye(mesh.spatial_dim) + G_phys
            Finv = np.linalg.inv(F)
            Ji = Jg_inv @ Finv
            if tangents is not None:
                tau = tangents[iq]
            else:
                nrm = np.asarray(level_set.gradient(x_phys), dtype=float)
                nrm /= (np.linalg.norm(nrm) + 1e-30)
                tau = np.array([-nrm[1], nrm[0]], dtype=float)
            stretch = float(np.linalg.norm(F @ tau))
            w_here = float(w) * stretch
            tau_norms.append(stretch)
        Ji_vals.append(Ji)
        w_eff.append(w_here)

    jit_geo = problem.dh.precompute_interface_factors(
        np.asarray([sample_eid], dtype=np.int32),
        qdeg,
        level_set,
        deformation=deformation,
        reuse=False,
    )
    if detail:
        print(f"    interface jit keys: {sorted(jit_geo.keys())}")
    jit_qw = jit_geo.get("qw", np.zeros((1, 0), dtype=float))
    jit_qw = jit_qw[0] if jit_qw.size else np.zeros(0, dtype=float)

    data = {
        "python_w": np.asarray(w_eff, dtype=float),
        "jit_w": np.asarray(jit_qw, dtype=float),
        "python_Ji": np.asarray(Ji_vals, dtype=float),
        "tau_norms": np.asarray(tau_norms, dtype=float) if tau_norms else None,
    }

    if detail:
        print(f"    interface weights sample_eid={sample_eid}, python_n={len(w_eff)}, jit_n={len(jit_qw)}")
        n_shared = min(len(w_eff), len(jit_qw))
        if n_shared:
            diff_w = data["python_w"][:n_shared] - data["jit_w"][:n_shared]
            print(
                f"    interface weights: max|python-jit|={np.max(np.abs(diff_w)):.3e}, "
                f"python_sum={np.sum(data['python_w']):.6e}, jit_sum={np.sum(data['jit_w']):.6e}"
            )
    return data

def geometry_report(problem: ProblemData, sample_eid: int, side: str, detail: bool = False) -> Dict[str, Dict[str, List[float]]]:
    mesh = problem.mesh
    if sample_eid < 0 or sample_eid >= mesh.n_elements:
        raise ValueError(f"sample_eid {sample_eid} outside mesh range (0..{mesh.n_elements - 1})")

    qdeg = problem.qvol
    py_geom = collect_python_geom(mesh, problem.deformation, problem.level_set, sample_eid, qdeg, side)
    jit_geom = collect_jit_geom(problem.dh, problem.level_set, problem.deformation, sample_eid, qdeg, side)

    # Align by point distance (tolerate reordering between implementations)
    data = {"python": [], "jit": []}

    for i, Ji in enumerate(py_geom["Ji"]):
        det = py_geom["det"][i]
        data["python"].append({
            "idx": i,
            "det": float(det),
            "Ji_flat": Ji.reshape(-1).tolist(),
        })

    for i, Ji in enumerate(jit_geom["Ji"]):
        det = jit_geom["det"][i]
        data["jit"].append({
            "idx": i,
            "det": float(det),
            "Ji_flat": Ji.reshape(-1).tolist(),
        })

    if detail:
        print(f"\nGeometry snapshot on element {sample_eid} (side {side}):")
        k = min(len(data["python"]), len(data["jit"]))
        for i in range(k):
            det_py = data["python"][i]["det"]
            det_jit = data["jit"][i]["det"]
            Ji_py = np.array(data["python"][i]["Ji_flat"]).reshape(2, 2)
            Ji_jit = np.array(data["jit"][i]["Ji_flat"]).reshape(2, 2)
            max_diff = float(np.max(np.abs(Ji_py - Ji_jit)))
            print(
                f"  qp {i:02d}: |det_py-det_jit|={abs(det_py - det_jit):.3e},"
                f" max|ΔJi|={max_diff:.3e}"
            )
    return data


def main():
    parser = argparse.ArgumentParser(description="Diagnose Stokes backend differences.")
    parser.add_argument("--with-deformation", action="store_true", help="use geometry deformation (default)")
    parser.add_argument("--no-deformation", action="store_true", help="disable deformation for control run")
    parser.add_argument("--detail", action="store_true", help="print per-term diagnostics")
    parser.add_argument("--sample-eid", type=int, default=-1, help="cut element id for geometry logging")
    parser.add_argument("--geom-side", choices=["+", "-"], default="+", help="side for cut-volume geometry check")
    parser.add_argument("--save-json", type=str, default="", help="optional path to dump raw comparison data")
    args = parser.parse_args()

    with_deformation = True
    if args.no_deformation:
        with_deformation = False
    if args.with_deformation:
        with_deformation = True

    problem = build_stokes_problem(with_deformation=with_deformation)

    python_contrib = assemble_integral_contributions(problem, backend="python")
    jit_contrib = assemble_integral_contributions(problem, backend="jit")

    report = compare_contributions(python_contrib, jit_contrib, detail=args.detail)

    if args.sample_eid < 0:
        cut_ids = problem.mesh.element_bitset("cut").to_indices()
        if len(cut_ids) == 0:
            raise RuntimeError("No cut elements in mesh; cannot perform geometry logging.")
        sample_eid = int(cut_ids[0])
    else:
        sample_eid = int(args.sample_eid)

    geom = geometry_report(problem, sample_eid, args.geom_side, detail=args.detail)
    iface_geom = interface_geometry_report(problem, sample_eid, problem.qvol + 2, detail=args.detail)

    print("\nSummary (python vs jit):")
    for part in ("lhs", "rhs"):
        if not report[part]:
            continue
        worst = max(report[part], key=lambda e: e["max_abs"])
        print(
            f"  {part.upper()} terms: {len(report[part])} entries,"
            f" worst max|Δ|={worst['max_abs']:.3e} (rel={worst['rel_fro']:.3e}) -> {worst['label']}"
        )

    if args.detail:
        try:
            idx_pen = next(i for i, t in enumerate(python_contrib['lhs']) if t.get('label') == 'ifc_penalty')
        except StopIteration:
            idx_pen = None
        if idx_pen is not None:
            K_py = python_contrib['lhs'][idx_pen]['matrix']
            K_jit = jit_contrib['lhs'][idx_pen]['matrix']
            K_py_dense = K_py.toarray() if hasattr(K_py, 'toarray') else np.asarray(K_py)
            K_jit_dense = K_jit.toarray() if hasattr(K_jit, 'toarray') else np.asarray(K_jit)
            diff = K_py_dense - K_jit_dense
            max_abs = float(np.max(np.abs(diff)))
            rel_norm = float(np.linalg.norm(diff) / max(np.linalg.norm(K_py_dense), 1e-14))
            diag_diff = np.diag(diff)
            nnz = int(np.count_nonzero(np.abs(diff) > 1e-12))
            max_idx = np.unravel_index(int(np.argmax(np.abs(diff))), diff.shape)
            py_val = K_py_dense[max_idx]
            jit_val = K_jit_dense[max_idx]
            print(f"    ifc_penalty detail: max|diff|={max_abs:.3e}, rel_norm={rel_norm:.3e}, nnz={nnz}")
            print(f"    ifc_penalty max entry @ {max_idx}: python={py_val:.6e}, jit={jit_val:.6e}")
            print(f"    ifc_penalty diag diff first 6 entries: {diag_diff[:6]}")

        if idx_pen is not None:
            geo_one = problem.dh.precompute_interface_factors(
                np.asarray([sample_eid], dtype=np.int32),
                problem.qvol + 2,
                problem.level_set,
                deformation=problem.deformation,
                reuse=False,
            )
            gdofs = geo_one['gdofs_map'][0].astype(int)
            const_pen = 440.0 / 0.125
            pos_masks, neg_masks = _hfa.build_side_masks_by_field(
                problem.dh, ['u_pos_x', 'u_pos_y', 'u_neg_x', 'u_neg_y'], sample_eid, problem.level_set, tol=SIDE.tol
            )
            if args.detail:
                print(f"    pos_masks keys={list(pos_masks.keys())}, neg_masks keys={list(neg_masks.keys())}")
                for name, mask in pos_masks.items():
                    print(f"      pos_mask {name}: {mask}")
                for name, mask in neg_masks.items():
                    print(f"      neg_mask {name}: {mask}")
            me = problem.dh.mixed_element
            local_manual = np.zeros((gdofs.size, gdofs.size), dtype=float)
            for q in range(geo_one['qw'].shape[1]):
                w = float(geo_one['qw'][0, q])
                neg_stack = np.stack((geo_one['b_u_neg_x'][0, q].copy(), geo_one['b_u_neg_y'][0, q].copy()))
                pos_stack = np.stack((geo_one['b_u_pos_x'][0, q].copy(), geo_one['b_u_pos_y'][0, q].copy()))
                sx = me.component_dof_slices['u_pos_x']
                sy = me.component_dof_slices['u_pos_y']
                pos_stack[0, sx] *= pos_masks.get('u_pos_x', np.ones(sx.stop - sx.start))
                pos_stack[1, sy] *= pos_masks.get('u_pos_y', np.ones(sy.stop - sy.start))
                sxn = me.component_dof_slices['u_neg_x']
                syn = me.component_dof_slices['u_neg_y']
                neg_stack[0, sxn] *= neg_masks.get('u_neg_x', np.ones(sxn.stop - sxn.start))
                neg_stack[1, syn] *= neg_masks.get('u_neg_y', np.ones(syn.stop - syn.start))
                jump = neg_stack - pos_stack
                local_manual += const_pen * (jump.T @ jump) * w
            K_py_local = K_py_dense[np.ix_(gdofs, gdofs)]
            K_jit_local = K_jit_dense[np.ix_(gdofs, gdofs)]
            print(f"    ifc_penalty local diag (python) first 6: {np.diag(K_py_local)[:6]}")
            print(f"    ifc_penalty local diag (jit)    first 6: {np.diag(K_jit_local)[:6]}")
            print(f"    ifc_penalty local diag (manual) first 6: {np.diag(local_manual)[:6]}")

    print(
        f"  Geometry sample element {sample_eid}:"
        f" python qpts={len(geom['python'])}, jit qpts={len(geom['jit'])}"
    )

    if args.save_json:
        payload = {
            "with_deformation": with_deformation,
            "report": report,
            "geometry": geom,
        }
        with open(args.save_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Saved diagnostics to {args.save_json}")


if __name__ == "__main__":
    main()
