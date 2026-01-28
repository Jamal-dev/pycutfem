"""FPI MMS Example 4.1 with *two meshes* and non-matching Nitsche coupling.

This driver mirrors the paper setup:
  - Ω^F discretized on a rotated (45°) background quad mesh, CutFEM for the
    truncated inlet (x=x0) and the hole around Ω^P.
  - Ω^P discretized on its own rotated (30°) body-fitted quad mesh.
  - Γ^FP coupled via non-matching Nitsche terms integrated on the CutFEM
    interface segments of the fluid mesh (no reliance on "aligned interface edges").

Scope (debug-first)
-------------------
This is a correctness/debug driver. It uses a simple coupled Newton loop and
assembles:
  - fluid volume terms via UFL + CutFEM dx on the fluid mesh,
  - poro volume terms via UFL on the poro mesh,
  - interface terms via explicit quadrature (python) on Γ^FP.

The goal is to check whether the *geometric/setup* mismatch was the root cause
of the non-paper convergence trend.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet, MinLevelSet, RotatedBoxLevelSet, ScaledLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.nonmatching.fpi_cutfem_nitsche import (
    assemble_fpi_interface_nitsche,
    assemble_inlet_traction_rhs,
    build_interface_from_cutfem_segments,
)
from pycutfem.nonmatching.system import apply_dirichlet_increment, coupled_dirichlet_data
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    FacetNormal,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
    div,
    dot,
    grad,
    inner,
    jump,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dCutSkeleton, dx
from pycutfem.ufl.analytic import Analytic
from pycutfem.utils.fpi_mms_example41 import build_example41_mms
from pycutfem.utils.fpi_poro_eulerian import jacobian_poro, residual_poro
from pycutfem.utils.fsi_fully_eulerian import build_measures, make_domain_sets
from pycutfem.utils.meshgen import structured_quad


def _epsilon(v):
    return Constant(0.5) * (grad(v) + grad(v).T)


def _approx_vinf(mms, bbox, n: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(int(seed))
    x0, x1, y0, y1 = map(float, bbox)
    X = np.empty((n, 2), dtype=float)
    X[:, 0] = rng.uniform(x0, x1, size=n)
    X[:, 1] = rng.uniform(y0, y1, size=n)
    V = np.asarray(mms.vF_k(X[:, 0], X[:, 1]), dtype=float)
    return float(np.max(np.linalg.norm(V, axis=1)))


@dataclass(frozen=True)
class TwoMeshProblem:
    mesh_f: Mesh
    mesh_p: Mesh
    dh_f: DofHandler
    dh_p: DofHandler
    fluid_ls: object
    poro_ls: object
    cut_ls: object
    domains_f: dict
    dx_f: object
    dG_f: object
    iface_fp: object
    inlet: dict


def build_two_mesh_problem(*, nx_f: int, nx_p: int, poly_order: int, qdeg: int, x0: float = -0.45) -> TwoMeshProblem:
    poro_ls = RotatedBoxLevelSet(center=(0.0, 0.0), hx=0.25, hy=0.25, angle=math.pi / 6.0)

    # --- fluid mesh: rotated 45° square mesh of size 1 ---
    nodes_f, elems_f, edges_f, corners_f = structured_quad(
        1.0,
        1.0,
        nx=nx_f,
        ny=nx_f,
        poly_order=poly_order,
        offset=(-0.5, -0.5),
        rotation=math.pi / 4.0,
        rotation_center=(0.0, 0.0),
    )
    mesh_f = Mesh(nodes_f, elems_f, edges_f, corners_f, element_type="quad", poly_order=poly_order)

    cut_ls = AffineLevelSet(-1.0, 0.0, float(x0))  # negative on {x>=x0}
    cut_pos = ScaledLevelSet(-1.0, cut_ls)  # positive on {x>=x0}
    fluid_ls = MinLevelSet(poro_ls, cut_pos)  # positive in Ω^F

    mesh_f.classify_elements(fluid_ls)
    mesh_f.classify_edges(fluid_ls)
    mesh_f.build_interface_segments(fluid_ls)

    # Boundary tags for Dirichlet on the *matching* outer boundary (portion with x>=x0).
    def _tag_paper_outer_boundaries():
        tol = 1e-10
        mesh_f.tag_boundary_edges(
            {
                "outer_dirichlet": lambda x, y: float(cut_ls(np.array([x, y], dtype=float))) <= tol,
                "outer_removed": lambda x, y: float(cut_ls(np.array([x, y], dtype=float))) > tol,
            }
        )

    _tag_paper_outer_boundaries()

    domains_f = make_domain_sets(mesh_f, use_aligned_interface=False)
    dx_f, _dx_inactive, _dGamma, dG_f, _dG_inactive = build_measures(mesh_f, fluid_ls, domains_f, qvol=int(qdeg))

    # --- poro mesh: rotated 30° square mesh of size 0.5 ---
    nodes_p, elems_p, edges_p, corners_p = structured_quad(
        0.5,
        0.5,
        nx=nx_p,
        ny=nx_p,
        poly_order=poly_order,
        offset=(-0.25, -0.25),
        rotation=math.pi / 6.0,
        rotation_center=(0.0, 0.0),
    )
    mesh_p = Mesh(nodes_p, elems_p, edges_p, corners_p, element_type="quad", poly_order=poly_order)
    mesh_p.build_grid_search()

    # DofHandlers
    me_f = MixedElement(mesh_f, {"v_pos_x": poly_order, "v_pos_y": poly_order, "p_pos_": poly_order})
    dh_f = DofHandler(me_f, method="cg")

    me_p = MixedElement(mesh_p, {"v_neg_x": poly_order, "v_neg_y": poly_order, "u_neg_x": poly_order, "u_neg_y": poly_order, "p_neg_": poly_order})
    dh_p = DofHandler(me_p, method="cg")

    # Tag inactive fluid dofs (anything supported only on "inside" elements)
    dh_f.dof_tags["inactive"] = set()
    for fld in ("v_pos_x", "v_pos_y", "p_pos_"):
        dh_f.tag_dofs_from_element_bitset("inactive", fld, "inside", strict=True)

    # Build interface segments for Γ^FP and inlet segments for Γ^{F,N}
    iface_fp, inlet = build_interface_from_cutfem_segments(
        mesh_f=mesh_f,
        fluid_ls=fluid_ls,
        poro_ls=poro_ls,
        mesh_p=mesh_p,
        x0=float(x0),
    )

    return TwoMeshProblem(
        mesh_f=mesh_f,
        mesh_p=mesh_p,
        dh_f=dh_f,
        dh_p=dh_p,
        fluid_ls=fluid_ls,
        poro_ls=poro_ls,
        cut_ls=cut_ls,
        domains_f=domains_f,
        dx_f=dx_f,
        dG_f=dG_f,
        iface_fp=iface_fp,
        inlet=inlet,
    )


def _grad_inner_jump(u, v, n):
    return inner(jump(grad(u), n), jump(grad(v), n))


def _set_vec_function_from_U(v: VectorFunction, dh: DofHandler, U: np.ndarray, fields: list[str]) -> None:
    for fld in fields:
        sl = np.asarray(dh.get_field_slice(fld), dtype=int)
        v.set_nodal_values(sl, np.asarray(U, float)[sl])


def _set_scalar_function_from_U(f: Function, dh: DofHandler, U: np.ndarray, field: str) -> None:
    sl = np.asarray(dh.get_field_slice(field), dtype=int)
    f.set_nodal_values(sl, np.asarray(U, float)[sl])


def _assemble_fluid_volume(
    prob: TwoMeshProblem,
    *,
    Uf_k: np.ndarray,
    Uf_n: np.ndarray,
    dt: float,
    rho_f: float,
    mu_f: float,
    qdeg: int,
    backend: str = "python",
    fF: object | None = None,
    use_stabilization: bool = True,
    gamma_u: float = 0.05,
    gamma_p: float = 0.05,
    gamma_div_factor: float = 1.0e-3,
    gamma_gp_nu: float = 0.1,
    gamma_gp_t: float = 0.001,
):
    dh = prob.dh_f
    dx_f = prob.dx_f
    dG_f = prob.dG_f

    # Spaces and (trial/test) on fluid mesh
    V = FunctionSpace("Vf", ["v_pos_x", "v_pos_y"])
    Q = FunctionSpace("Qf", ["p_pos_"])
    dv = VectorTrialFunction(space=V, dof_handler=dh)
    dp = TrialFunction(field_name="p_pos_", name="dp", dof_handler=dh)
    w = VectorTestFunction(space=V, dof_handler=dh)
    q = TestFunction(field_name="p_pos_", name="q", dof_handler=dh)

    v_k = VectorFunction(name="v_k", field_names=["v_pos_x", "v_pos_y"], dof_handler=dh)
    v_n = VectorFunction(name="v_n", field_names=["v_pos_x", "v_pos_y"], dof_handler=dh)
    p_k = Function(name="p_k", field_name="p_pos_", dof_handler=dh)
    p_n = Function(name="p_n", field_name="p_pos_", dof_handler=dh)

    _set_vec_function_from_U(v_k, dh, Uf_k, ["v_pos_x", "v_pos_y"])
    _set_vec_function_from_U(v_n, dh, Uf_n, ["v_pos_x", "v_pos_y"])
    _set_scalar_function_from_U(p_k, dh, Uf_k, "p_pos_")
    _set_scalar_function_from_U(p_n, dh, Uf_n, "p_pos_")

    vdot = (v_k - v_n) / Constant(float(dt))
    th = Constant(1.0)
    r = inner(Constant(rho_f) * vdot, w) * dx_f
    r += th * Constant(rho_f) * dot(dot(grad(v_k), v_k), w) * dx_f
    r += (Constant(1.0) - th) * Constant(rho_f) * dot(dot(grad(v_n), v_n), w) * dx_f
    r += Constant(2.0) * th * Constant(mu_f) * inner(_epsilon(v_k), _epsilon(w)) * dx_f
    r += Constant(2.0) * (Constant(1.0) - th) * Constant(mu_f) * inner(_epsilon(v_n), _epsilon(w)) * dx_f
    r += -p_k * div(w) * dx_f + q * div(v_k) * dx_f
    if fF is not None:
        r += -dot(fF, w) * dx_f

    a = Constant(rho_f) / Constant(float(dt)) * dot(dv, w) * dx_f
    a += th * Constant(rho_f) * dot(dot(grad(v_k), dv), w) * dx_f
    a += th * Constant(rho_f) * dot(dot(grad(dv), v_k), w) * dx_f
    a += Constant(2.0) * th * Constant(mu_f) * inner(_epsilon(dv), _epsilon(w)) * dx_f
    a += -dp * div(w) * dx_f + q * div(dv) * dx_f

    if use_stabilization:
        n = FacetNormal()
        cell_h = CellDiameter()
        gamma_div = float(gamma_div_factor) * float(gamma_p)

        derivs = {(0, 1), (1, 0)}
        dS_f = dCutSkeleton(level_set=prob.fluid_ls, metadata={"q": int(qdeg), "side": "+", "derivs": derivs})

        # Lightweight CIP + ghost penalty stabilization (paper eqs. (21)-(23) patterns).
        r += Constant(float(gamma_p)) * (cell_h**3) * _grad_inner_jump(p_k, q, n) * dS_f
        a += Constant(float(gamma_p)) * (cell_h**3) * _grad_inner_jump(dp, q, n) * dS_f

        r += Constant(float(gamma_div)) * cell_h * jump(div(v_k)) * jump(div(w)) * dS_f
        a += Constant(float(gamma_div)) * cell_h * jump(div(dv)) * jump(div(w)) * dS_f

        # Ghost penalties near the interface/cut cells.
        tau_gp_u = Constant(float(gamma_gp_nu)) * cell_h * Constant(float(mu_f)) + Constant(float(gamma_gp_t)) * (cell_h**3) * (
            Constant(float(rho_f)) / (th * Constant(float(dt)))
        )
        r += tau_gp_u * _grad_inner_jump(v_k, w, n) * dG_f
        a += tau_gp_u * _grad_inner_jump(dv, w, n) * dG_f

        r += Constant(float(gamma_p)) * (cell_h**3) * _grad_inner_jump(p_k, q, n) * dG_f
        a += Constant(float(gamma_p)) * (cell_h**3) * _grad_inner_jump(dp, q, n) * dG_f

    K, F = assemble_form(Equation(a, r), dof_handler=dh, bcs=[], quad_order=int(qdeg), backend=str(backend))
    return K.tocsr(), np.asarray(F, dtype=float)


def _assemble_poro_volume(
    prob: TwoMeshProblem,
    *,
    Up_k: np.ndarray,
    Up_n: np.ndarray,
    Up_nm1: np.ndarray,
    dt: float,
    rho_f: float,
    mu_f: float,
    rho_s0: float,
    porosity: float,
    qdeg: int,
    backend: str = "python",
    K_inv: np.ndarray,
    c_nh: float,
    beta_nh: float,
    f_mass: object | None = None,
    fD: object | None = None,
    fS: object | None = None,
):
    dh = prob.dh_p
    dx_p = dx(metadata={"q": int(qdeg)})

    # Spaces / trial-test
    Vp = FunctionSpace("Vp", ["v_neg_x", "v_neg_y"])
    Up = FunctionSpace("Up", ["u_neg_x", "u_neg_y"])
    Qp = FunctionSpace("Qp", ["p_neg_"])
    dvP = VectorTrialFunction(space=Vp, dof_handler=dh)
    duP = VectorTrialFunction(space=Up, dof_handler=dh)
    dpP = TrialFunction(field_name="p_neg_", name="dpP", dof_handler=dh)
    vP_test = VectorTestFunction(space=Vp, dof_handler=dh)
    uP_test = VectorTestFunction(space=Up, dof_handler=dh)
    qP_test = TestFunction(field_name="p_neg_", name="qP", dof_handler=dh)

    vP_k = VectorFunction(name="vP_k", field_names=["v_neg_x", "v_neg_y"], dof_handler=dh)
    uP_k = VectorFunction(name="uP_k", field_names=["u_neg_x", "u_neg_y"], dof_handler=dh)
    pP_k = Function(name="pP_k", field_name="p_neg_", dof_handler=dh)
    vP_n = VectorFunction(name="vP_n", field_names=["v_neg_x", "v_neg_y"], dof_handler=dh)
    uP_n = VectorFunction(name="uP_n", field_names=["u_neg_x", "u_neg_y"], dof_handler=dh)
    pP_n = Function(name="pP_n", field_name="p_neg_", dof_handler=dh)
    uP_nm1_f = VectorFunction(name="uP_nm1", field_names=["u_neg_x", "u_neg_y"], dof_handler=dh)

    _set_vec_function_from_U(vP_k, dh, Up_k, ["v_neg_x", "v_neg_y"])
    _set_vec_function_from_U(uP_k, dh, Up_k, ["u_neg_x", "u_neg_y"])
    _set_scalar_function_from_U(pP_k, dh, Up_k, "p_neg_")
    _set_vec_function_from_U(vP_n, dh, Up_n, ["v_neg_x", "v_neg_y"])
    _set_vec_function_from_U(uP_n, dh, Up_n, ["u_neg_x", "u_neg_y"])
    _set_scalar_function_from_U(pP_n, dh, Up_n, "p_neg_")
    _set_vec_function_from_U(uP_nm1_f, dh, Up_nm1, ["u_neg_x", "u_neg_y"])

    dtc = Constant(float(dt))
    th = Constant(1.0)
    rho_fc = Constant(float(rho_f))
    mu_fc = Constant(float(mu_f))
    rho_s = Constant(float(rho_s0))
    phi = Constant(float(porosity))
    Kinv = Constant(np.asarray(K_inv, float).tolist(), dim=2)
    cnh = Constant(float(c_nh))
    bnh = Constant(float(beta_nh))

    r = residual_poro(
        vP_k,
        uP_k,
        pP_k,
        vP_n,
        uP_n,
        pP_n,
        qP_test,
        vP_test,
        uP_test,
        u_nm1=uP_nm1_f,
        rho_f=rho_fc,
        mu_f=mu_fc,
        rho_s0_tilde=rho_s,
        phi=phi,
        K_inv=Kinv,
        c_nh=cnh,
        beta_nh=bnh,
        dt=dtc,
        theta=th,
        dx_p=dx_p,
    )
    if f_mass is not None:
        r += -qP_test * f_mass * dx_p
    if fD is not None:
        r += -dot(fD, vP_test) * dx_p
    if fS is not None:
        r += -dot(fS, uP_test) * dx_p
    a = jacobian_poro(
        vP_k,
        uP_k,
        pP_k,
        uP_n,
        dvP,
        duP,
        dpP,
        qP_test,
        vP_test,
        uP_test,
        u_nm1=uP_nm1_f,
        rho_f=rho_fc,
        mu_f=mu_fc,
        rho_s0_tilde=rho_s,
        phi=phi,
        K_inv=Kinv,
        c_nh=cnh,
        beta_nh=bnh,
        dt=dtc,
        theta=th,
        dx_p=dx_p,
    )
    K, F = assemble_form(Equation(a, r), dof_handler=dh, bcs=[], quad_order=int(qdeg), backend=str(backend))
    return K.tocsr(), np.asarray(F, dtype=float)


def _parse_nx_list(s: str) -> list[int]:
    s = str(s).strip()
    if not s:
        return []
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _eoc(err_prev: float, err: float, h_prev: float, h: float) -> float:
    if err_prev <= 0.0 or err <= 0.0 or h_prev <= 0.0 or h <= 0.0:
        return float("nan")
    return float(np.log(err / err_prev) / np.log(h / h_prev))


def _interface_trace_errors(
    *,
    interface,
    dh_f: DofHandler,
    dh_p: DofHandler,
    Uf: np.ndarray,
    Up: np.ndarray,
    mms,
    mu_f: float,
    quad_order: int,
) -> dict[str, float]:
    """Interface (Γ^FP) L2 errors for traces and fluid traction.

    We integrate over the CutFEM segment interface using the same segment geometry
    that drives the nonmatching coupling.
    """
    from pycutfem.fem import transform
    from pycutfem.integration.quadrature import gauss_legendre

    if interface.n_segments() <= 0:
        return {
            "err_vF_G": 0.0,
            "err_vP_G": 0.0,
            "err_pF_G": 0.0,
            "err_pP_G": 0.0,
            "err_tF_G": 0.0,
            "err_gsigma_n": 0.0,
        }

    mesh_f = dh_f.mixed_element.mesh
    mesh_p = dh_p.mixed_element.mesh
    me_f = dh_f.mixed_element
    me_p = dh_p.mixed_element

    vfx, vfy, pf = "v_pos_x", "v_pos_y", "p_pos_"
    vpx, vpy, upx, upy, pp = "v_neg_x", "v_neg_y", "u_neg_x", "u_neg_y", "p_neg_"

    xi_1d, w_1d = gauss_legendre(int(quad_order))
    xi_1d = np.asarray(xi_1d, dtype=float)
    w_1d = np.asarray(w_1d, dtype=float)

    sl_vfx = me_f.component_dof_slices[vfx]
    sl_vfy = me_f.component_dof_slices[vfy]
    sl_pf = me_f.component_dof_slices[pf]

    sl_vpx = me_p.component_dof_slices[vpx]
    sl_vpy = me_p.component_dof_slices[vpy]
    sl_upx = me_p.component_dof_slices[upx]
    sl_upy = me_p.component_dof_slices[upy]
    sl_pp = me_p.component_dof_slices[pp]

    mu_f = float(mu_f)
    I2 = np.eye(2, dtype=float)

    acc_vF2 = 0.0
    acc_vP2 = 0.0
    acc_pF2 = 0.0
    acc_pP2 = 0.0
    acc_tF2 = 0.0
    acc_gsig_n2 = 0.0

    for s in range(interface.n_segments()):
        ef = int(interface.pos_elem_ids[s])
        ep = int(interface.neg_elem_ids[s])

        n_vec = np.asarray(interface.n[s], dtype=float).reshape(2,)
        nF = -n_vec  # fluid outward

        P0 = np.asarray(interface.P0[s], float)
        P1 = np.asarray(interface.P1[s], float)
        mid = 0.5 * (P0 + P1)
        half = 0.5 * (P1 - P0)
        segJ = float(np.linalg.norm(half))
        if segJ <= 0.0:
            continue

        gd_vfx = np.asarray(dh_f.element_maps[vfx][ef], dtype=int)
        gd_vfy = np.asarray(dh_f.element_maps[vfy][ef], dtype=int)
        gd_pf = np.asarray(dh_f.element_maps[pf][ef], dtype=int)

        gd_vpx = np.asarray(dh_p.element_maps[vpx][ep], dtype=int)
        gd_vpy = np.asarray(dh_p.element_maps[vpy][ep], dtype=int)
        gd_upx = np.asarray(dh_p.element_maps[upx][ep], dtype=int)
        gd_upy = np.asarray(dh_p.element_maps[upy][ep], dtype=int)
        gd_pp = np.asarray(dh_p.element_maps[pp][ep], dtype=int)

        vf_x = np.asarray(Uf, float)[gd_vfx]
        vf_y = np.asarray(Uf, float)[gd_vfy]
        pf_c = np.asarray(Uf, float)[gd_pf]

        vp_x = np.asarray(Up, float)[gd_vpx]
        vp_y = np.asarray(Up, float)[gd_vpy]
        up_x = np.asarray(Up, float)[gd_upx]
        up_y = np.asarray(Up, float)[gd_upy]
        pp_c = np.asarray(Up, float)[gd_pp]
        _ = (up_x, up_y)  # reserved (may be used for additional diagnostics)

        for q, wq_ref in enumerate(w_1d):
            xq = mid + float(xi_1d[q]) * half
            wq = float(wq_ref) * segJ
            if wq == 0.0:
                continue

            xi_f, eta_f = transform.inverse_mapping(mesh_f, ef, xq)
            xi_p, eta_p = transform.inverse_mapping(mesh_p, ep, xq)

            Jf = transform.jacobian(mesh_f, ef, (float(xi_f), float(eta_f)))
            Jp = transform.jacobian(mesh_p, ep, (float(xi_p), float(eta_p)))
            Jif = np.linalg.inv(Jf)
            _ = np.linalg.inv(Jp)  # Jip (not needed yet)

            phi_vfx = np.asarray(me_f.basis(vfx, float(xi_f), float(eta_f)), float).ravel()[sl_vfx]
            phi_vfy = np.asarray(me_f.basis(vfy, float(xi_f), float(eta_f)), float).ravel()[sl_vfy]
            phi_pf = np.asarray(me_f.basis(pf, float(xi_f), float(eta_f)), float).ravel()[sl_pf]
            g_vfx = (np.asarray(me_f.grad_basis(vfx, float(xi_f), float(eta_f)), float) @ Jif)[sl_vfx]
            g_vfy = (np.asarray(me_f.grad_basis(vfy, float(xi_f), float(eta_f)), float) @ Jif)[sl_vfy]

            phi_vpx = np.asarray(me_p.basis(vpx, float(xi_p), float(eta_p)), float).ravel()[sl_vpx]
            phi_vpy = np.asarray(me_p.basis(vpy, float(xi_p), float(eta_p)), float).ravel()[sl_vpy]
            phi_pp = np.asarray(me_p.basis(pp, float(xi_p), float(eta_p)), float).ravel()[sl_pp]

            vF = np.array([float(vf_x @ phi_vfx), float(vf_y @ phi_vfy)], dtype=float)
            pF = float(pf_c @ phi_pf)
            vP = np.array([float(vp_x @ phi_vpx), float(vp_y @ phi_vpy)], dtype=float)
            pP = float(pp_c @ phi_pp)

            dvx = np.array([float(vf_x @ g_vfx[:, 0]), float(vf_x @ g_vfx[:, 1])], dtype=float)
            dvy = np.array([float(vf_y @ g_vfy[:, 0]), float(vf_y @ g_vfy[:, 1])], dtype=float)
            grad_vF = np.array([[dvx[0], dvx[1]], [dvy[0], dvy[1]]], dtype=float)
            eps = 0.5 * (grad_vF + grad_vF.T)
            sigmaF_h = -pF * I2 + (2.0 * mu_f) * eps
            tractionF_h = sigmaF_h @ nF

            x, y = float(xq[0]), float(xq[1])
            vF_A = np.asarray(mms.vF_k(x, y), float).reshape(2,)
            vP_A = np.asarray(mms.vP_k(x, y), float).reshape(2,)
            pF_A = float(mms.pF_k(x, y))
            pP_A = float(mms.pP_k(x, y))
            sigmaF_A = np.asarray(mms.sigmaF_k(x, y), float)
            tractionF_A = np.asarray(sigmaF_A @ nF, float).reshape(2,)

            dvF = vF - vF_A
            dvP = vP - vP_A
            dpF = pF - pF_A
            dpP = pP - pP_A
            dtF = tractionF_h - tractionF_A

            acc_vF2 += wq * float(dvF @ dvF)
            acc_vP2 += wq * float(dvP @ dvP)
            acc_pF2 += wq * float(dpF * dpF)
            acc_pP2 += wq * float(dpP * dpP)
            acc_tF2 += wq * float(dtF @ dtF)

            # scalar quantity used in the paper via g_sigma_n:
            # visc_n = nF · (σF nF) + pP
            gsig_n_A = float((nF @ tractionF_A) + pP_A)
            visc_n_h = float((nF @ tractionF_h) + pP)
            dgs = visc_n_h - gsig_n_A
            acc_gsig_n2 += wq * float(dgs * dgs)

    return {
        "err_vF_G": float(math.sqrt(max(float(acc_vF2), 0.0))),
        "err_vP_G": float(math.sqrt(max(float(acc_vP2), 0.0))),
        "err_pF_G": float(math.sqrt(max(float(acc_pF2), 0.0))),
        "err_pP_G": float(math.sqrt(max(float(acc_pP2), 0.0))),
        "err_tF_G": float(math.sqrt(max(float(acc_tF2), 0.0))),
        "err_gsigma_n": float(math.sqrt(max(float(acc_gsig_n2), 0.0))),
    }


def _fluid_domain_l2_errors(
    *,
    prob: TwoMeshProblem,
    Uf: np.ndarray,
    mms,
    quad_order: int,
    backend: str,
) -> dict[str, float]:
    """L2 errors on Ω^F using the CutFEM `dx_f` measure (correct domain)."""
    from pycutfem.ufl.forms import Equation, assemble_form

    dh = prob.dh_f
    dx_f = prob.dx_f

    vF_h = VectorFunction(name="vF_h", field_names=["v_pos_x", "v_pos_y"], dof_handler=dh)
    pF_h = Function(name="pF_h", field_name="p_pos_", dof_handler=dh)
    _set_vec_function_from_U(vF_h, dh, Uf, ["v_pos_x", "v_pos_y"])
    _set_scalar_function_from_U(pF_h, dh, Uf, "p_pos_")

    vF_A = Analytic(lambda x, y: mms.vF_k(x, y), degree=int(quad_order))
    pF_A = Analytic(lambda x, y: mms.pF_k(x, y), degree=int(quad_order))

    e_vF2 = dot(vF_h - vF_A, vF_h - vF_A)
    e_pF2 = (pF_h - pF_A) * (pF_h - pF_A)

    L = e_vF2 * dx_f + e_pF2 * dx_f
    scalars = assemble_form(
        Equation(None, L),
        dof_handler=dh,
        bcs=[],
        quad_order=int(quad_order),
        assembler_hooks={
            e_vF2: {"name": "e_vF2"},
            e_pF2: {"name": "e_pF2"},
        },
        backend=str(backend),
    )
    ev2 = float(np.asarray(scalars["e_vF2"]).reshape(()))
    ep2 = float(np.asarray(scalars["e_pF2"]).reshape(()))
    return {
        "err_vF": float(math.sqrt(max(ev2, 0.0))),
        "err_pF": float(math.sqrt(max(ep2, 0.0))),
    }


def _inlet_trace_errors(
    *,
    prob: TwoMeshProblem,
    Uf: np.ndarray,
    mms,
    mu_f: float,
    quad_order: int,
) -> dict[str, float]:
    """Inlet (Γ^{F,N}) L2 errors for vF/pF and traction σ_F nF."""
    from pycutfem.fem import transform
    from pycutfem.integration.quadrature import gauss_legendre

    inlet_P0 = np.asarray(prob.inlet.get("inlet_P0", np.zeros((0, 2))), float)
    inlet_P1 = np.asarray(prob.inlet.get("inlet_P1", np.zeros((0, 2))), float)
    inlet_pos_ids = np.asarray(prob.inlet.get("inlet_pos_elem_ids", np.zeros((0,), dtype=int)), dtype=int)
    if inlet_P0.shape[0] == 0:
        return {"err_vF_in": 0.0, "err_pF_in": 0.0, "err_tF_in": 0.0}

    dh = prob.dh_f
    mesh_f = dh.mixed_element.mesh
    me_f = dh.mixed_element
    vfx, vfy, pf = "v_pos_x", "v_pos_y", "p_pos_"
    sl_vfx = me_f.component_dof_slices[vfx]
    sl_vfy = me_f.component_dof_slices[vfy]
    sl_pf = me_f.component_dof_slices[pf]

    xi_1d, w_1d = gauss_legendre(int(quad_order))
    xi_1d = np.asarray(xi_1d, dtype=float)
    w_1d = np.asarray(w_1d, dtype=float)

    mu_f = float(mu_f)
    I2 = np.eye(2, dtype=float)
    nF = np.array([-1.0, 0.0], dtype=float)

    acc_v2 = 0.0
    acc_p2 = 0.0
    acc_t2 = 0.0

    for s in range(int(inlet_P0.shape[0])):
        ef = int(inlet_pos_ids[s])
        P0 = np.asarray(inlet_P0[s], float)
        P1 = np.asarray(inlet_P1[s], float)
        mid = 0.5 * (P0 + P1)
        half = 0.5 * (P1 - P0)
        segJ = float(np.linalg.norm(half))
        if segJ <= 0.0:
            continue

        gd_vfx = np.asarray(dh.element_maps[vfx][ef], dtype=int)
        gd_vfy = np.asarray(dh.element_maps[vfy][ef], dtype=int)
        gd_pf = np.asarray(dh.element_maps[pf][ef], dtype=int)

        vf_x = np.asarray(Uf, float)[gd_vfx]
        vf_y = np.asarray(Uf, float)[gd_vfy]
        pf_c = np.asarray(Uf, float)[gd_pf]

        for q, wq_ref in enumerate(w_1d):
            xq = mid + float(xi_1d[q]) * half
            wq = float(wq_ref) * segJ
            if wq == 0.0:
                continue

            xi_f, eta_f = transform.inverse_mapping(mesh_f, ef, xq)
            Jf = transform.jacobian(mesh_f, ef, (float(xi_f), float(eta_f)))
            Jif = np.linalg.inv(Jf)

            phi_vfx = np.asarray(me_f.basis(vfx, float(xi_f), float(eta_f)), float).ravel()[sl_vfx]
            phi_vfy = np.asarray(me_f.basis(vfy, float(xi_f), float(eta_f)), float).ravel()[sl_vfy]
            phi_pf = np.asarray(me_f.basis(pf, float(xi_f), float(eta_f)), float).ravel()[sl_pf]
            g_vfx = (np.asarray(me_f.grad_basis(vfx, float(xi_f), float(eta_f)), float) @ Jif)[sl_vfx]
            g_vfy = (np.asarray(me_f.grad_basis(vfy, float(xi_f), float(eta_f)), float) @ Jif)[sl_vfy]

            vF = np.array([float(vf_x @ phi_vfx), float(vf_y @ phi_vfy)], dtype=float)
            pF = float(pf_c @ phi_pf)

            dvx = np.array([float(vf_x @ g_vfx[:, 0]), float(vf_x @ g_vfx[:, 1])], dtype=float)
            dvy = np.array([float(vf_y @ g_vfy[:, 0]), float(vf_y @ g_vfy[:, 1])], dtype=float)
            grad_vF = np.array([[dvx[0], dvx[1]], [dvy[0], dvy[1]]], dtype=float)
            eps = 0.5 * (grad_vF + grad_vF.T)
            sigmaF_h = -pF * I2 + (2.0 * mu_f) * eps
            tractionF_h = sigmaF_h @ nF

            x, y = float(xq[0]), float(xq[1])
            vF_A = np.asarray(mms.vF_k(x, y), float).reshape(2,)
            pF_A = float(mms.pF_k(x, y))
            sigmaF_A = np.asarray(mms.sigmaF_k(x, y), float)
            tractionF_A = np.asarray(sigmaF_A @ nF, float).reshape(2,)

            dv = vF - vF_A
            dp = pF - pF_A
            dt = tractionF_h - tractionF_A
            acc_v2 += wq * float(dv @ dv)
            acc_p2 += wq * float(dp * dp)
            acc_t2 += wq * float(dt @ dt)

    return {
        "err_vF_in": float(math.sqrt(max(float(acc_v2), 0.0))),
        "err_pF_in": float(math.sqrt(max(float(acc_p2), 0.0))),
        "err_tF_in": float(math.sqrt(max(float(acc_t2), 0.0))),
    }


def _save_convergence_plots(*, outdir: Path, rows_by_iface: dict[str, list[dict]], backend: str, p: int, iface_arg: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] matplotlib unavailable; skipping convergence plots ({exc})")
        return

    outdir.mkdir(parents=True, exist_ok=True)

    colors = {"bj": "C0", "bjs": "C1"}
    iface_keys = [k for k in ("bj", "bjs") if k in rows_by_iface]

    metrics: list[tuple[str, str, float]] = [
        ("err_vF", "e(vF)", 2.0),
        ("err_pF", "e(pF)", 2.0),
        ("err_vP", "e(vP)", 2.0),
        ("err_uP", "e(uP)", 2.0),
        ("err_pP", "e(pP)", 2.0),
        ("err_vF_in", "e(vF)_inlet", 1.5),
        ("err_pF_in", "e(pF)_inlet", 1.5),
        ("err_tF_in", "e(tF)_inlet", 0.5),
        ("err_vF_G", "e(vF)_Γ", 1.5),
        ("err_vP_G", "e(vP)_Γ", 1.5),
        ("err_pF_G", "e(pF)_Γ", 1.5),
        ("err_pP_G", "e(pP)_Γ", 1.5),
        ("err_tF_G", "e(tF)_Γ", 0.5),
        ("err_gsigma_n", "e(g_sigma_n)", 0.5),
    ]

    for key, title, slope_ref in metrics:
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        any_series = False

        for iface in iface_keys:
            rows = rows_by_iface.get(iface, [])
            hs = np.asarray([float(r["h"]) for r in rows], dtype=float)
            ys = np.asarray([float(r.get(key, float("nan"))) for r in rows], dtype=float)
            ok = np.isfinite(hs) & np.isfinite(ys) & (hs > 0.0) & (ys > 0.0)
            if not np.any(ok):
                continue
            any_series = True
            ax.loglog(hs[ok], ys[ok], "o-", label=iface.upper(), color=colors.get(iface, None))

        if not any_series:
            plt.close(fig)
            continue

        # reference line anchored at the finest h among all plotted series
        h_all = []
        y_all = []
        for iface in iface_keys:
            rows = rows_by_iface.get(iface, [])
            for r in rows:
                h = float(r["h"])
                y = float(r.get(key, float("nan")))
                if h > 0.0 and y > 0.0 and np.isfinite(y):
                    h_all.append(h)
                    y_all.append(y)
        if h_all:
            h_all = np.asarray(h_all, float)
            y_all = np.asarray(y_all, float)
            i = int(np.argmin(h_all))
            h0 = float(h_all[i])
            y0 = float(y_all[i])
            href = np.array([h0, float(np.max(h_all))], dtype=float)
            yref = y0 * (href / h0) ** float(slope_ref)
            ax.loglog(href, yref, "--", color="0.35", alpha=0.7, label=f"h^{slope_ref:g} ref")

        ax.set_xlabel("h")
        ax.set_ylabel(title)
        ax.grid(True, which="both", ls=":", alpha=0.35)
        ax.legend(loc="best", fontsize=9)

        outpath = outdir / f"two_mesh_example41_{key}_backend-{backend}_p{int(p)}_iface-{iface_arg}.png"
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)


def _save_mesh_plots(*, prob: TwoMeshProblem, outdir: Path, nx_f: int, nx_p: int, p: int, tag: str) -> None:
    import matplotlib.pyplot as plt

    from pycutfem.io.visualization import plot_mesh_2

    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 9))
    # Use element tags + interface segments instead of a green φ=0 contour
    # to avoid tricontour artifacts when φ=0 at boundary nodes.
    plot_mesh_2(
        prob.mesh_f,
        show=False,
        ax=ax,
        plot_nodes=False,
        plot_interface=True,
        elem_tags=True,
        edge_colors=True,
        fluid_solid_overlay=False,
    )
    ax.set_title("Fluid mesh (CutFEM): elem tags + ghost/interface edges + segments")
    fig.savefig(outdir / f"mesh_fluid_nx{nx_f}_p{p}_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_mesh_2(
        prob.mesh_p,
        show=False,
        ax=ax,
        plot_nodes=False,
        plot_interface=False,
        elem_tags=True,
        edge_colors=True,
        fluid_solid_overlay=False,
    )
    ax.set_title("Poro mesh (body-fitted)")
    fig.savefig(outdir / f"mesh_poro_nx{nx_p}_p{p}_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _run_one(
    *,
    nx_f: int,
    poly_order: int,
    qdeg: int,
    dt: float,
    x0: float,
    backend: str,
    interface: str,
    newton_it: int,
    newton_tol: float,
    check_only: bool,
    save_mesh: bool,
    outdir: Path,
    use_stabilization: bool,
) -> dict[str, float]:
    nx_f = int(nx_f)
    nx_p = max(1, nx_f // 2)

    prob = build_two_mesh_problem(nx_f=nx_f, nx_p=nx_p, poly_order=int(poly_order), qdeg=int(qdeg), x0=float(x0))

    interface = str(interface).strip().lower()
    if interface not in {"bj", "bjs"}:
        raise ValueError("interface must be one of {'bj','bjs'}")
    beta_BJ = 1.0 if interface == "bj" else 0.0

    # MMS (use same builder as single-mesh driver, with rotated interface)
    poro_ls = prob.poro_ls
    interface_name = "rotated_box"
    interface_params = (float(poro_ls.center[0]), float(poro_ls.center[1]), float(poro_ls.hx), float(poro_ls.hy), float(poro_ls.angle))
    mms = build_example41_mms(dt_val=float(dt), kinv_case="iso", t_prev=0.0, beta_BJ=float(beta_BJ), interface=interface_name, interface_params=interface_params)

    # parameters (paper)
    rho_f = 1.0
    mu_f = 1.0
    porosity = 0.5
    K = 0.10
    K_inv = (1.0 / K) * np.eye(2, dtype=float)
    E = 1000.0
    nu = 0.30
    mu_s = E / (2.0 * (1.0 + nu))
    c_nh = mu_s / 2.0
    beta_nh = nu / (1.0 - 2.0 * nu)
    kappa = math.sqrt(K) / (1.0 * mu_f * math.sqrt(porosity))

    # Nitsche/penalty parameters (keep as in the debug driver)
    gamma_inv = 45.0
    gamma = 1.0 / gamma_inv
    zeta = -1.0

    bbox = (
        float(prob.mesh_f.nodes_x_y_pos[:, 0].min()),
        float(prob.mesh_f.nodes_x_y_pos[:, 0].max()),
        float(prob.mesh_f.nodes_x_y_pos[:, 1].min()),
        float(prob.mesh_f.nodes_x_y_pos[:, 1].max()),
    )
    v_inf = _approx_vinf(mms, bbox=bbox)

    # Initial conditions at t=0 (exact), current guess at t=dt (exact)
    dt = float(dt)

    # Fluid vectors (full DOF vector)
    Uf_n = np.zeros(int(prob.dh_f.total_dofs), dtype=float)
    Uf_k = np.zeros_like(Uf_n)

    dof_xy_v = prob.dh_f.get_dof_coords("v_pos_x")
    dof_xy_p = prob.dh_f.get_dof_coords("p_pos_")
    vF_n = np.asarray(mms.vF_n(dof_xy_v[:, 0], dof_xy_v[:, 1]), float)
    vF_k = np.asarray(mms.vF_k(dof_xy_v[:, 0], dof_xy_v[:, 1]), float)
    Uf_n[prob.dh_f.get_field_slice("v_pos_x")] = vF_n[:, 0]
    Uf_n[prob.dh_f.get_field_slice("v_pos_y")] = vF_n[:, 1]
    Uf_n[prob.dh_f.get_field_slice("p_pos_")] = np.asarray(mms.pF_n(dof_xy_p[:, 0], dof_xy_p[:, 1]), float).reshape(-1)
    Uf_k[prob.dh_f.get_field_slice("v_pos_x")] = vF_k[:, 0]
    Uf_k[prob.dh_f.get_field_slice("v_pos_y")] = vF_k[:, 1]
    Uf_k[prob.dh_f.get_field_slice("p_pos_")] = np.asarray(mms.pF_k(dof_xy_p[:, 0], dof_xy_p[:, 1]), float).reshape(-1)

    # Poro vectors
    Up_n = np.zeros(int(prob.dh_p.total_dofs), dtype=float)
    Up_nm1 = np.zeros_like(Up_n)
    Up_k = np.zeros_like(Up_n)
    dof_xy = {f: prob.dh_p.get_dof_coords(f) for f in ("v_neg_x", "v_neg_y", "u_neg_x", "u_neg_y", "p_neg_")}
    vP_n = np.asarray(mms.vP_n(dof_xy["v_neg_x"][:, 0], dof_xy["v_neg_x"][:, 1]), float)
    uP_n = np.asarray(mms.uP_n(dof_xy["u_neg_x"][:, 0], dof_xy["u_neg_x"][:, 1]), float)
    Up_n[prob.dh_p.get_field_slice("v_neg_x")] = vP_n[:, 0]
    Up_n[prob.dh_p.get_field_slice("v_neg_y")] = vP_n[:, 1]
    Up_n[prob.dh_p.get_field_slice("u_neg_x")] = uP_n[:, 0]
    Up_n[prob.dh_p.get_field_slice("u_neg_y")] = uP_n[:, 1]
    Up_n[prob.dh_p.get_field_slice("p_neg_")] = np.asarray(mms.pP_n(dof_xy["p_neg_"][:, 0], dof_xy["p_neg_"][:, 1]), float).reshape(-1)
    Up_nm1[:] = Up_n

    vP_k = np.asarray(mms.vP_k(dof_xy["v_neg_x"][:, 0], dof_xy["v_neg_x"][:, 1]), float)
    uP_k = np.asarray(mms.uP_k(dof_xy["u_neg_x"][:, 0], dof_xy["u_neg_x"][:, 1]), float)
    Up_k[prob.dh_p.get_field_slice("v_neg_x")] = vP_k[:, 0]
    Up_k[prob.dh_p.get_field_slice("v_neg_y")] = vP_k[:, 1]
    Up_k[prob.dh_p.get_field_slice("u_neg_x")] = uP_k[:, 0]
    Up_k[prob.dh_p.get_field_slice("u_neg_y")] = uP_k[:, 1]
    Up_k[prob.dh_p.get_field_slice("p_neg_")] = np.asarray(mms.pP_k(dof_xy["p_neg_"][:, 0], dof_xy["p_neg_"][:, 1]), float).reshape(-1)

    # inlet traction RHS uses σF_A evaluated at quadrature points
    def _sigmaF_A(X):
        X = np.asarray(X, float)
        return np.asarray(mms.sigmaF_k(X[:, 0], X[:, 1]), float)

    # Interface jumps (xq is (2,), nF is (2,))
    def g_sigma(xq, nF):
        xq = np.asarray(xq, float).reshape(2,)
        nF = np.asarray(nF, float).reshape(2,)
        sigF = np.asarray(mms.sigmaF_k(float(xq[0]), float(xq[1])), float)
        sigP = np.asarray(mms.sigmaP_k(float(xq[0]), float(xq[1])), float)
        return np.asarray((sigF - sigP) @ nF, float).reshape(2,)

    def g_sigma_n(xq, nF):
        xq = np.asarray(xq, float).reshape(2,)
        nF = np.asarray(nF, float).reshape(2,)
        sigF = np.asarray(mms.sigmaF_k(float(xq[0]), float(xq[1])), float)
        tF = sigF @ nF
        pP = float(mms.pP_k(float(xq[0]), float(xq[1])))
        return float(np.dot(nF, tF) + pP)

    def g_n(xq, nF):
        xq = np.asarray(xq, float).reshape(2,)
        vF = np.asarray(mms.vF_k(float(xq[0]), float(xq[1])), float).reshape(2,)
        vP = np.asarray(mms.vP_k(float(xq[0]), float(xq[1])), float).reshape(2,)
        u1 = np.asarray(mms.uP_k(float(xq[0]), float(xq[1])), float).reshape(2,)
        u0 = np.asarray(mms.uP_n(float(xq[0]), float(xq[1])), float).reshape(2,)
        u_dot = (u1 - u0) / float(dt)
        return np.asarray(vF - u_dot - float(porosity) * (vP - u_dot), float).reshape(2,)

    def g_t(xq, nF):
        xq = np.asarray(xq, float).reshape(2,)
        nF = np.asarray(nF, float).reshape(2,)
        sigF = np.asarray(mms.sigmaF_k(float(xq[0]), float(xq[1])), float)
        tF = sigF @ nF
        vF = np.asarray(mms.vF_k(float(xq[0]), float(xq[1])), float).reshape(2,)
        vP = np.asarray(mms.vP_k(float(xq[0]), float(xq[1])), float).reshape(2,)
        u1 = np.asarray(mms.uP_k(float(xq[0]), float(xq[1])), float).reshape(2,)
        u0 = np.asarray(mms.uP_n(float(xq[0]), float(xq[1])), float).reshape(2,)
        u_dot = (u1 - u0) / float(dt)
        return np.asarray(vF - u_dot - float(beta_BJ) * float(porosity) * (vP - u_dot) + float(kappa) * tF, float).reshape(2,)

    # MMS forcing (volume)
    fF = Analytic(lambda x, y: mms.fF(x, y), degree=int(qdeg))
    fD = Analytic(lambda x, y: mms.fD(x, y), degree=int(qdeg))
    fS = Analytic(lambda x, y: mms.fS(x, y), degree=int(qdeg))
    f_mass = Analytic(lambda x, y: np.asarray(mms.f_mass(x, y), float), degree=int(qdeg))

    # Dirichlet BCs on matching outer boundary (fluid velocity).
    bcs_f = [
        BoundaryCondition("v_pos_x", "dirichlet", "outer_dirichlet", mms.vF_x),
        BoundaryCondition("v_pos_y", "dirichlet", "outer_dirichlet", mms.vF_y),
    ]
    data_dirichlet = coupled_dirichlet_data(
        dh_pos=prob.dh_f,
        bcs_pos=bcs_f,
        dh_neg=prob.dh_p,
        bcs_neg=[],
        neg_offset=int(prob.dh_f.total_dofs),
    )
    for gd in prob.dh_f.dof_tags.get("inactive", set()):
        data_dirichlet[int(gd)] = 0.0

    # Pin one pressure DOF on each mesh to remove the constant-pressure nullspace.
    def _pin_pressure(dh: DofHandler, field: str, *, offset: int, exact_cb):
        sl = np.asarray(dh.get_field_slice(field), dtype=int)
        if sl.size == 0:
            return
        coords = np.asarray(dh.get_dof_coords(field), float)
        gd = int(sl[0]) + int(offset)
        x, y = float(coords[0, 0]), float(coords[0, 1])
        data_dirichlet[gd] = float(exact_cb(x, y))

    _pin_pressure(prob.dh_f, "p_pos_", offset=0, exact_cb=lambda x, y: float(mms.pF_k(x, y)))
    _pin_pressure(prob.dh_p, "p_neg_", offset=int(prob.dh_f.total_dofs), exact_cb=lambda x, y: float(mms.pP_k(x, y)))

    def _print_block_norms(
        tag: str,
        dh: DofHandler,
        R: np.ndarray,
        fields: list[str],
        *,
        offset: int = 0,
        constrained: np.ndarray | None = None,
    ):
        print(f"\n{tag}")
        for fld in fields:
            sl = np.asarray(dh.get_field_slice(fld), dtype=int) + int(offset)
            if constrained is not None and sl.size:
                sl = sl[~np.asarray(constrained[sl], dtype=bool)]
            val = float(np.linalg.norm(R[sl], ord=np.inf)) if sl.size else 0.0
            print(f"  |R[{fld}]|_inf = {val:.3e}")

    # Newton loop on coupled system
    U = np.concatenate([Uf_k.copy(), Up_k.copy()])

    Uf = U[: int(prob.dh_f.total_dofs)]
    Up = U[int(prob.dh_f.total_dofs) :]

    if check_only:
        Kf, Rf = _assemble_fluid_volume(
            prob,
            Uf_k=Uf,
            Uf_n=Uf_n,
            dt=dt,
            rho_f=rho_f,
            mu_f=mu_f,
            qdeg=int(qdeg),
            backend=str(backend),
            fF=fF,
            use_stabilization=bool(use_stabilization),
        )
        Kp, Rp = _assemble_poro_volume(
            prob,
            Up_k=Up,
            Up_n=Up_n,
            Up_nm1=Up_nm1,
            dt=dt,
            rho_f=rho_f,
            mu_f=mu_f,
            rho_s0=1.0,
            porosity=porosity,
            qdeg=int(qdeg),
            backend=str(backend),
            K_inv=K_inv,
            c_nh=c_nh,
            beta_nh=beta_nh,
            f_mass=f_mass,
            fD=fD,
            fS=fS,
        )

        Kif, Rif = assemble_fpi_interface_nitsche(
            interface=prob.iface_fp,
            dh_f=prob.dh_f,
            dh_p=prob.dh_p,
            Uf=Uf,
            Up=Up,
            Up_n=Up_n,
            dt=dt,
            rho_f=rho_f,
            mu_f=mu_f,
            porosity=porosity,
            beta_BJ=beta_BJ,
            kappa=kappa,
            gamma_n=gamma,
            gamma_t=gamma,
            zeta=zeta,
            vF_inf=v_inf,
            c_v_gamma=1.0 / 6.0,
            c_t_gamma=1.0 / 12.0,
            quad_order=int(qdeg),
            g_sigma=g_sigma,
            g_sigma_n=g_sigma_n,
            g_n=g_n,
            g_t=g_t,
        )

        Rin = assemble_inlet_traction_rhs(
            inlet_P0=prob.inlet["inlet_P0"],
            inlet_P1=prob.inlet["inlet_P1"],
            inlet_pos_elem_ids=prob.inlet["inlet_pos_elem_ids"],
            dh_f=prob.dh_f,
            sigmaF_A=_sigmaF_A,
            quad_order=int(qdeg),
            x0=float(x0),
        )

        K = sp.bmat([[Kf, None], [None, Kp]], format="csr")
        R = np.concatenate([Rf + Rin, Rp])
        K = (K + Kif).tocsr()
        R = R + Rif

        constrained = np.zeros(int(R.size), dtype=bool)
        rows = np.fromiter((int(k) for k in data_dirichlet.keys()), dtype=int)
        if rows.size:
            constrained[rows] = True

        _print_block_norms("Volume residual (free dofs)", prob.dh_f, R, ["v_pos_x", "v_pos_y", "p_pos_"], offset=0, constrained=constrained)
        _print_block_norms(
            "Volume residual (free dofs)",
            prob.dh_p,
            R,
            ["v_neg_x", "v_neg_y", "u_neg_x", "u_neg_y", "p_neg_"],
            offset=int(prob.dh_f.total_dofs),
            constrained=constrained,
        )
        _print_block_norms("Interface residual (free dofs)", prob.dh_f, Rif, ["v_pos_x", "v_pos_y", "p_pos_"], offset=0, constrained=constrained)
        _print_block_norms(
            "Interface residual (free dofs)",
            prob.dh_p,
            Rif,
            ["v_neg_x", "v_neg_y", "u_neg_x", "u_neg_y", "p_neg_"],
            offset=int(prob.dh_f.total_dofs),
            constrained=constrained,
        )
        return {
            "h": 1.0 / float(nx_f),
            "err_vF": float("nan"),
            "err_pF": float("nan"),
            "err_vP": float("nan"),
            "err_uP": float("nan"),
            "err_pP": float("nan"),
        }

    for it in range(int(newton_it)):
        Uf = U[: int(prob.dh_f.total_dofs)]
        Up = U[int(prob.dh_f.total_dofs) :]

        Kf, Rf = _assemble_fluid_volume(
            prob,
            Uf_k=Uf,
            Uf_n=Uf_n,
            dt=dt,
            rho_f=rho_f,
            mu_f=mu_f,
            qdeg=int(qdeg),
            backend=str(backend),
            fF=fF,
            use_stabilization=bool(use_stabilization),
        )
        Kp, Rp = _assemble_poro_volume(
            prob,
            Up_k=Up,
            Up_n=Up_n,
            Up_nm1=Up_nm1,
            dt=dt,
            rho_f=rho_f,
            mu_f=mu_f,
            rho_s0=1.0,
            porosity=porosity,
            qdeg=int(qdeg),
            backend=str(backend),
            K_inv=K_inv,
            c_nh=c_nh,
            beta_nh=beta_nh,
            f_mass=f_mass,
            fD=fD,
            fS=fS,
        )

        Kif, Rif = assemble_fpi_interface_nitsche(
            interface=prob.iface_fp,
            dh_f=prob.dh_f,
            dh_p=prob.dh_p,
            Uf=Uf,
            Up=Up,
            Up_n=Up_n,
            dt=dt,
            rho_f=rho_f,
            mu_f=mu_f,
            porosity=porosity,
            beta_BJ=beta_BJ,
            kappa=kappa,
            gamma_n=gamma,
            gamma_t=gamma,
            zeta=zeta,
            vF_inf=v_inf,
            c_v_gamma=1.0 / 6.0,
            c_t_gamma=1.0 / 12.0,
            quad_order=int(qdeg),
            g_sigma=g_sigma,
            g_sigma_n=g_sigma_n,
            g_n=g_n,
            g_t=g_t,
        )

        Rin = assemble_inlet_traction_rhs(
            inlet_P0=prob.inlet["inlet_P0"],
            inlet_P1=prob.inlet["inlet_P1"],
            inlet_pos_elem_ids=prob.inlet["inlet_pos_elem_ids"],
            dh_f=prob.dh_f,
            sigmaF_A=_sigmaF_A,
            quad_order=int(qdeg),
            x0=float(x0),
        )

        K = sp.bmat([[Kf, None], [None, Kp]], format="csr")
        R = np.concatenate([Rf + Rin, Rp])
        K = (K + Kif).tocsr()
        R = R + Rif

        constrained = np.zeros(int(R.size), dtype=bool)
        rows = np.fromiter((int(k) for k in data_dirichlet.keys()), dtype=int)
        if rows.size:
            constrained[rows] = True
        res = float(np.linalg.norm(R[~constrained], ord=np.inf)) if np.any(~constrained) else 0.0
        print(f"[nx={nx_f:4d} iface={interface.upper():3s}] Newton {it+1}: |R_free|_inf = {res:.3e}")
        if res < float(newton_tol):
            break

        Kc, rhs = apply_dirichlet_increment(K, R, U, data_dirichlet)
        dU = spla.spsolve(Kc, rhs)
        U = U + np.asarray(dU, float)

    # Error summary -----------------------------------------------------------
    # IMPORTANT: For CutFEM fields, report errors on Ω^F using the CutFEM dx_f
    # measure (not a full-element quadrature over the background mesh).
    errs_f = _fluid_domain_l2_errors(
        prob=prob,
        Uf=U[: int(prob.dh_f.total_dofs)],
        mms=mms,
        quad_order=int(qdeg),
        backend=str(backend),
    )
    err_vF = float(errs_f["err_vF"])
    err_pF = float(errs_f["err_pF"])
    err_vP = prob.dh_p.l2_error(
        U[int(prob.dh_f.total_dofs) :],
        exact={
            "v_neg_x": lambda x, y: float(np.asarray(mms.vP_k(x, y), float)[0]),
            "v_neg_y": lambda x, y: float(np.asarray(mms.vP_k(x, y), float)[1]),
        },
        quad_order=int(qdeg),
        relative=False,
    )
    err_uP = prob.dh_p.l2_error(
        U[int(prob.dh_f.total_dofs) :],
        exact={
            "u_neg_x": lambda x, y: float(np.asarray(mms.uP_k(x, y), float)[0]),
            "u_neg_y": lambda x, y: float(np.asarray(mms.uP_k(x, y), float)[1]),
        },
        quad_order=int(qdeg),
        relative=False,
    )
    err_pP = prob.dh_p.l2_error(
        U[int(prob.dh_f.total_dofs) :],
        exact={"p_neg_": mms.pP_k},
        quad_order=int(qdeg),
        relative=False,
    )

    inlet_errs = _inlet_trace_errors(
        prob=prob,
        Uf=U[: int(prob.dh_f.total_dofs)],
        mms=mms,
        mu_f=mu_f,
        quad_order=int(qdeg),
    )
    ifc = _interface_trace_errors(
        interface=prob.iface_fp,
        dh_f=prob.dh_f,
        dh_p=prob.dh_p,
        Uf=U[: int(prob.dh_f.total_dofs)],
        Up=U[int(prob.dh_f.total_dofs) :],
        mms=mms,
        mu_f=mu_f,
        quad_order=int(qdeg),
    )

    h = 1.0 / float(nx_f)
    print(
        f"[nx={nx_f:4d} iface={interface.upper():3s}] h~{h:.3e}  |e(vF)|={err_vF:.3e}  |e(pF)|={err_pF:.3e}  |e(vP)|={err_vP:.3e}  |e(uP)|={err_uP:.3e}  |e(pP)|={err_pP:.3e}  |e(vF)_in|={inlet_errs['err_vF_in']:.3e}  |e(pF)_in|={inlet_errs['err_pF_in']:.3e}  |e(tF)_in|={inlet_errs['err_tF_in']:.3e}  |e(vF)_Γ|={ifc['err_vF_G']:.3e}  |e(pF)_Γ|={ifc['err_pF_G']:.3e}  |e(tF)_Γ|={ifc['err_tF_G']:.3e}"
    )

    if save_mesh:
        _save_mesh_plots(prob=prob, outdir=outdir, nx_f=nx_f, nx_p=nx_p, p=int(poly_order), tag=f"{interface}_backend-{backend}")

    return {
        "h": float(h),
        "err_vF": float(err_vF),
        "err_pF": float(err_pF),
        "err_vP": float(err_vP),
        "err_uP": float(err_uP),
        "err_pP": float(err_pP),
        **inlet_errs,
        **ifc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--q", type=int, default=6)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--x0", type=float, default=-0.45)
    parser.add_argument("--backend", type=str, default="cpp", choices=["python", "jit", "cpp"])
    parser.add_argument("--newton-it", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--newton-tol", type=float, default=0.0, help="Overrides --tol when > 0.")
    parser.add_argument("--interface", type=str, default="bj", choices=["bj", "bjs", "both"])
    parser.add_argument("--convergence", action="store_true", help="Run an h-refinement study (prints a table + saves plots).")
    parser.add_argument("--levels", type=int, default=5, help="Number of refinement levels (convergence mode).")
    parser.add_argument("--nx-list", type=str, default="", help="In convergence mode: comma-separated nx list (overrides --nx/--levels).")
    parser.add_argument(
        "--paper-h-range",
        action="store_true",
        help="In convergence mode: use the paper's h-range [0.25, 0.00390625] with 12 points (nx=[4,6,8,12,16,24,32,48,64,96,128,256]).",
    )
    parser.add_argument("--check-only", action="store_true", help="Assemble once at the initial guess and print residual norms, then exit.")
    parser.add_argument("--save-mesh", action="store_true", help="Save labeled mesh plots per refinement (PNG) into --outdir.")
    parser.add_argument("--plot", action="store_true", help="Deprecated alias for --save-mesh.")
    parser.add_argument("--no-stabilization", action="store_true", help="Disable CIP + ghost penalties (debug only).")
    parser.add_argument("--outdir", type=str, default="examples/FPI/_mms_example41_twomesh")
    args = parser.parse_args()

    newton_tol = float(args.newton_tol) if float(args.newton_tol) > 0.0 else float(args.tol)
    save_mesh = bool(args.save_mesh or args.plot)
    use_stabilization = not bool(args.no_stabilization)
    outdir = Path(str(args.outdir))

    if args.convergence:
        outdir.mkdir(parents=True, exist_ok=True)
        if str(args.nx_list).strip():
            nx_list = _parse_nx_list(str(args.nx_list))
        elif bool(args.paper_h_range):
            nx_list = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256]
        else:
            nx0 = int(args.nx)
            levels = int(args.levels)
            nx_list = [int(nx0 * (2**k)) for k in range(max(1, levels))]

        iface_list = [str(args.interface)] if str(args.interface) in {"bj", "bjs"} else ["bj", "bjs"]
        rows_by_iface: dict[str, list[dict]] = {}
        for iface in iface_list:
            rows: list[dict] = []
            for nx_i in nx_list:
                rows.append(
                    _run_one(
                        nx_f=int(nx_i),
                        poly_order=int(args.p),
                        qdeg=int(args.q),
                        dt=float(args.dt),
                        x0=float(args.x0),
                        backend=str(args.backend),
                        interface=str(iface),
                        newton_it=int(args.newton_it),
                        newton_tol=float(newton_tol),
                        check_only=False,
                        save_mesh=bool(save_mesh),
                        outdir=outdir,
                        use_stabilization=bool(use_stabilization),
                    )
                )
            rows_by_iface[str(iface)] = rows

            print(f"\nFPI Example 4.1 (two-mesh, nonmatching) | backend={args.backend} | p={args.p} | interface={iface.upper()}")
            print(f"{'h':>10} |e(vF)|    eoc |e(pF)|    eoc |e(vP)|    eoc |e(uP)|    eoc |e(pP)|    eoc")
            prev = None
            for r in rows:
                if prev is None:
                    print(
                        f"{r['h']:10.3e} {r['err_vF']:11.3e}    -  {r['err_pF']:11.3e}    -  {r['err_vP']:11.3e}    -  {r['err_uP']:11.3e}    -  {r['err_pP']:11.3e}    -"
                    )
                else:
                    print(
                        f"{r['h']:10.3e} {r['err_vF']:11.3e} { _eoc(prev['err_vF'], r['err_vF'], prev['h'], r['h']):5.2f}  {r['err_pF']:11.3e} { _eoc(prev['err_pF'], r['err_pF'], prev['h'], r['h']):5.2f}  {r['err_vP']:11.3e} { _eoc(prev['err_vP'], r['err_vP'], prev['h'], r['h']):5.2f}  {r['err_uP']:11.3e} { _eoc(prev['err_uP'], r['err_uP'], prev['h'], r['h']):5.2f}  {r['err_pP']:11.3e} { _eoc(prev['err_pP'], r['err_pP'], prev['h'], r['h']):5.2f}"
                    )
                prev = r

        _save_convergence_plots(
            outdir=outdir,
            rows_by_iface=rows_by_iface,
            backend=str(args.backend),
            p=int(args.p),
            iface_arg=str(args.interface),
        )
        return

    iface = str(args.interface) if str(args.interface) in {"bj", "bjs"} else "bj"
    _run_one(
        nx_f=int(args.nx),
        poly_order=int(args.p),
        qdeg=int(args.q),
        dt=float(args.dt),
        x0=float(args.x0),
        backend=str(args.backend),
        interface=str(iface),
        newton_it=int(args.newton_it),
        newton_tol=float(newton_tol),
        check_only=bool(args.check_only),
        save_mesh=bool(save_mesh),
        outdir=outdir,
        use_stabilization=bool(use_stabilization),
    )
    return

    nx_f = int(args.nx)
    nx_p = max(1, nx_f // 2)
    prob = build_two_mesh_problem(nx_f=nx_f, nx_p=nx_p, poly_order=int(args.p), qdeg=int(args.q), x0=float(args.x0))

    # MMS (use same builder as single-mesh driver, with rotated interface)
    poro_ls = prob.poro_ls
    interface_name = "rotated_box"
    interface_params = (float(poro_ls.center[0]), float(poro_ls.center[1]), float(poro_ls.hx), float(poro_ls.hy), float(poro_ls.angle))
    mms = build_example41_mms(dt_val=float(args.dt), kinv_case="iso", t_prev=0.0, beta_BJ=1.0, interface=interface_name, interface_params=interface_params)

    # parameters
    rho_f = 1.0
    mu_f = 1.0
    porosity = 0.5
    K = 0.10
    K_inv = (1.0 / K) * np.eye(2, dtype=float)
    E = 1000.0
    nu = 0.30
    mu_s = E / (2.0 * (1.0 + nu))
    c_nh = mu_s / 2.0
    beta_nh = nu / (1.0 - 2.0 * nu)
    beta_BJ = 1.0
    kappa = math.sqrt(K) / (1.0 * mu_f * math.sqrt(porosity))
    gamma_inv = 45.0
    gamma = 1.0 / gamma_inv
    zeta = -1.0

    bbox = (
        float(prob.mesh_f.nodes_x_y_pos[:, 0].min()),
        float(prob.mesh_f.nodes_x_y_pos[:, 0].max()),
        float(prob.mesh_f.nodes_x_y_pos[:, 1].min()),
        float(prob.mesh_f.nodes_x_y_pos[:, 1].max()),
    )
    v_inf = _approx_vinf(mms, bbox=bbox)

    # Initial conditions at t=0 (exact), current guess at t=dt (exact)
    dt = float(args.dt)
    t0 = 0.0
    t1 = dt

    # Fluid vectors (full DOF vector)
    Uf_n = np.zeros(int(prob.dh_f.total_dofs), dtype=float)
    Uf_k = np.zeros_like(Uf_n)
    # Fill by interpolation using nodal coords
    xy_f = np.asarray(prob.mesh_f.nodes_x_y_pos, float)
    # For CG spaces, dof coords match nodes for p=1; keep it simple: evaluate at dof coords
    dof_xy_vx = prob.dh_f.get_dof_coords("v_pos_x")
    dof_xy_vy = prob.dh_f.get_dof_coords("v_pos_y")
    dof_xy_p = prob.dh_f.get_dof_coords("p_pos_")
    vF_n = np.asarray(mms.vF_n(dof_xy_vx[:, 0], dof_xy_vx[:, 1]), float)
    vF_k = np.asarray(mms.vF_k(dof_xy_vx[:, 0], dof_xy_vx[:, 1]), float)
    Uf_n[prob.dh_f.get_field_slice("v_pos_x")] = vF_n[:, 0]
    Uf_n[prob.dh_f.get_field_slice("v_pos_y")] = vF_n[:, 1]
    Uf_n[prob.dh_f.get_field_slice("p_pos_")] = np.asarray(mms.pF_n(dof_xy_p[:, 0], dof_xy_p[:, 1]), float).reshape(-1)
    Uf_k[prob.dh_f.get_field_slice("v_pos_x")] = vF_k[:, 0]
    Uf_k[prob.dh_f.get_field_slice("v_pos_y")] = vF_k[:, 1]
    Uf_k[prob.dh_f.get_field_slice("p_pos_")] = np.asarray(mms.pF_k(dof_xy_p[:, 0], dof_xy_p[:, 1]), float).reshape(-1)

    # Poro vectors
    Up_n = np.zeros(int(prob.dh_p.total_dofs), dtype=float)
    Up_nm1 = np.zeros_like(Up_n)
    Up_k = np.zeros_like(Up_n)
    dof_xy = {f: prob.dh_p.get_dof_coords(f) for f in ("v_neg_x", "v_neg_y", "u_neg_x", "u_neg_y", "p_neg_")}
    vP_n = np.asarray(mms.vP_n(dof_xy["v_neg_x"][:, 0], dof_xy["v_neg_x"][:, 1]), float)
    uP_n = np.asarray(mms.uP_n(dof_xy["u_neg_x"][:, 0], dof_xy["u_neg_x"][:, 1]), float)
    Up_n[prob.dh_p.get_field_slice("v_neg_x")] = vP_n[:, 0]
    Up_n[prob.dh_p.get_field_slice("v_neg_y")] = vP_n[:, 1]
    Up_n[prob.dh_p.get_field_slice("u_neg_x")] = uP_n[:, 0]
    Up_n[prob.dh_p.get_field_slice("u_neg_y")] = uP_n[:, 1]
    Up_n[prob.dh_p.get_field_slice("p_neg_")] = np.asarray(mms.pP_n(dof_xy["p_neg_"][:, 0], dof_xy["p_neg_"][:, 1]), float).reshape(-1)
    Up_nm1[:] = Up_n

    vP_k = np.asarray(mms.vP_k(dof_xy["v_neg_x"][:, 0], dof_xy["v_neg_x"][:, 1]), float)
    uP_k = np.asarray(mms.uP_k(dof_xy["u_neg_x"][:, 0], dof_xy["u_neg_x"][:, 1]), float)
    Up_k[prob.dh_p.get_field_slice("v_neg_x")] = vP_k[:, 0]
    Up_k[prob.dh_p.get_field_slice("v_neg_y")] = vP_k[:, 1]
    Up_k[prob.dh_p.get_field_slice("u_neg_x")] = uP_k[:, 0]
    Up_k[prob.dh_p.get_field_slice("u_neg_y")] = uP_k[:, 1]
    Up_k[prob.dh_p.get_field_slice("p_neg_")] = np.asarray(mms.pP_k(dof_xy["p_neg_"][:, 0], dof_xy["p_neg_"][:, 1]), float).reshape(-1)

    # Manufactured interface data callables (xq is (2,), nF is (2,))
    def _sigmaF_A(X):
        X = np.asarray(X, float)
        return np.asarray(mms.sigmaF_k(X[:, 0], X[:, 1]), float)

    def _sigmaP_A(X):
        X = np.asarray(X, float)
        return np.asarray(mms.sigmaP_k(X[:, 0], X[:, 1]), float)

    def _pP_A(X):
        X = np.asarray(X, float)
        return np.asarray(mms.pP_k(X[:, 0], X[:, 1]), float).reshape(-1)

    def _vF_A(X):
        X = np.asarray(X, float)
        return np.asarray(mms.vF_k(X[:, 0], X[:, 1]), float)

    def _vP_A(X):
        X = np.asarray(X, float)
        return np.asarray(mms.vP_k(X[:, 0], X[:, 1]), float)

    def _u_dot_A(X):
        X = np.asarray(X, float)
        u1 = np.asarray(mms.uP_k(X[:, 0], X[:, 1]), float)
        u0 = np.asarray(mms.uP_n(X[:, 0], X[:, 1]), float)
        return (u1 - u0) / float(dt)

    def g_sigma(xq, nF):
        # Use the *discrete* interface normal provided by the segment integration
        # (paper Remark 8) instead of the smooth level-set gradient used by the
        # MMS builder.
        xq = np.asarray(xq, float).reshape(2,)
        nF = np.asarray(nF, float).reshape(2,)
        sigF = np.asarray(mms.sigmaF_k(float(xq[0]), float(xq[1])), float)
        sigP = np.asarray(mms.sigmaP_k(float(xq[0]), float(xq[1])), float)
        return np.asarray((sigF - sigP) @ nF, float).reshape(2,)

    def g_sigma_n(xq, nF):
        xq = np.asarray(xq, float).reshape(2,)
        nF = np.asarray(nF, float).reshape(2,)
        sigF = np.asarray(mms.sigmaF_k(float(xq[0]), float(xq[1])), float)
        tF = sigF @ nF
        pP = float(mms.pP_k(float(xq[0]), float(xq[1])))
        return float(np.dot(nF, tF) + pP)

    def g_n(xq, nF):
        xq = np.asarray(xq, float).reshape(2,)
        vF = np.asarray(mms.vF_k(float(xq[0]), float(xq[1])), float).reshape(2,)
        vP = np.asarray(mms.vP_k(float(xq[0]), float(xq[1])), float).reshape(2,)
        u1 = np.asarray(mms.uP_k(float(xq[0]), float(xq[1])), float).reshape(2,)
        u0 = np.asarray(mms.uP_n(float(xq[0]), float(xq[1])), float).reshape(2,)
        u_dot = (u1 - u0) / float(dt)
        return np.asarray(vF - u_dot - float(porosity) * (vP - u_dot), float).reshape(2,)

    def g_t(xq, nF):
        xq = np.asarray(xq, float).reshape(2,)
        nF = np.asarray(nF, float).reshape(2,)
        sigF = np.asarray(mms.sigmaF_k(float(xq[0]), float(xq[1])), float)
        tF = sigF @ nF
        vF = np.asarray(mms.vF_k(float(xq[0]), float(xq[1])), float).reshape(2,)
        vP = np.asarray(mms.vP_k(float(xq[0]), float(xq[1])), float).reshape(2,)
        u1 = np.asarray(mms.uP_k(float(xq[0]), float(xq[1])), float).reshape(2,)
        u0 = np.asarray(mms.uP_n(float(xq[0]), float(xq[1])), float).reshape(2,)
        u_dot = (u1 - u0) / float(dt)
        return np.asarray(vF - u_dot - float(beta_BJ) * float(porosity) * (vP - u_dot) + float(kappa) * tF, float).reshape(2,)

    # MMS forcing (volume)
    fF = Analytic(lambda x, y: mms.fF(x, y), degree=int(args.q))
    fD = Analytic(lambda x, y: mms.fD(x, y), degree=int(args.q))
    fS = Analytic(lambda x, y: mms.fS(x, y), degree=int(args.q))
    f_mass = Analytic(lambda x, y: np.asarray(mms.f_mass(x, y), float), degree=int(args.q))

    # Dirichlet BCs on matching outer boundary (fluid velocity).
    bcs_f = [
        BoundaryCondition("v_pos_x", "dirichlet", "outer_dirichlet", mms.vF_x),
        BoundaryCondition("v_pos_y", "dirichlet", "outer_dirichlet", mms.vF_y),
    ]
    data_dirichlet = coupled_dirichlet_data(
        dh_pos=prob.dh_f,
        bcs_pos=bcs_f,
        dh_neg=prob.dh_p,
        bcs_neg=[],
        neg_offset=int(prob.dh_f.total_dofs),
    )
    # Inactive fluid dofs -> homogeneous Dirichlet
    for gd in prob.dh_f.dof_tags.get("inactive", set()):
        data_dirichlet[int(gd)] = 0.0

    # Pin one pressure DOF on each mesh to remove the constant-pressure nullspace.
    # (The paper setup has traction at the inlet and interface coupling, but in
    # the discrete two-mesh system we enforce a unique reference explicitly.)
    def _pin_pressure(dh: DofHandler, field: str, *, offset: int, exact_cb):
        sl = np.asarray(dh.get_field_slice(field), dtype=int)
        if sl.size == 0:
            return
        coords = np.asarray(dh.get_dof_coords(field), float)
        # pick the first DOF
        gd = int(sl[0]) + int(offset)
        x, y = float(coords[0, 0]), float(coords[0, 1])
        data_dirichlet[gd] = float(exact_cb(x, y))

    _pin_pressure(prob.dh_f, "p_pos_", offset=0, exact_cb=lambda x, y: float(mms.pF_k(x, y)))
    _pin_pressure(prob.dh_p, "p_neg_", offset=int(prob.dh_f.total_dofs), exact_cb=lambda x, y: float(mms.pP_k(x, y)))

    def _print_block_norms(
        tag: str,
        dh: DofHandler,
        R: np.ndarray,
        fields: list[str],
        *,
        offset: int = 0,
        constrained: np.ndarray | None = None,
    ):
        print(f"\n{tag}")
        for fld in fields:
            sl = np.asarray(dh.get_field_slice(fld), dtype=int) + int(offset)
            if constrained is not None and sl.size:
                sl = sl[~np.asarray(constrained[sl], dtype=bool)]
            val = float(np.linalg.norm(R[sl], ord=np.inf)) if sl.size else 0.0
            print(f"  |R[{fld}]|_inf = {val:.3e}")

    # Newton loop on coupled system
    U = np.concatenate([Uf_k.copy(), Up_k.copy()])
    U_prev = np.concatenate([Uf_n.copy(), Up_n.copy()])

    if args.check_only:
        Uf = U[: int(prob.dh_f.total_dofs)]
        Up = U[int(prob.dh_f.total_dofs) :]
        Kf, Rf = _assemble_fluid_volume(prob, Uf_k=Uf, Uf_n=Uf_n, dt=dt, rho_f=rho_f, mu_f=mu_f, qdeg=int(args.q), fF=fF)
        Kp, Rp = _assemble_poro_volume(
            prob,
            Up_k=Up,
            Up_n=Up_n,
            Up_nm1=Up_nm1,
            dt=dt,
            rho_f=rho_f,
            mu_f=mu_f,
            rho_s0=1.0,
            porosity=porosity,
            qdeg=int(args.q),
            K_inv=K_inv,
            c_nh=c_nh,
            beta_nh=beta_nh,
            f_mass=f_mass,
            fD=fD,
            fS=fS,
        )
        Kif, Rif = assemble_fpi_interface_nitsche(
            interface=prob.iface_fp,
            dh_f=prob.dh_f,
            dh_p=prob.dh_p,
            Uf=Uf,
            Up=Up,
            Up_n=Up_n,
            dt=dt,
            rho_f=rho_f,
            mu_f=mu_f,
            porosity=porosity,
            beta_BJ=beta_BJ,
            kappa=kappa,
            gamma_n=gamma,
            gamma_t=gamma,
            zeta=zeta,
            vF_inf=v_inf,
            c_v_gamma=1.0 / 6.0,
            c_t_gamma=1.0 / 12.0,
            quad_order=int(args.q),
            g_sigma=g_sigma,
            g_sigma_n=g_sigma_n,
            g_n=g_n,
            g_t=g_t,
        )
        Rin = assemble_inlet_traction_rhs(
            inlet_P0=prob.inlet["inlet_P0"],
            inlet_P1=prob.inlet["inlet_P1"],
            inlet_pos_elem_ids=prob.inlet["inlet_pos_elem_ids"],
            dh_f=prob.dh_f,
            sigmaF_A=_sigmaF_A,
            quad_order=int(args.q),
            x0=float(args.x0),
        )
        R = np.concatenate([Rf + Rin, Rp]) + Rif
        K = sp.bmat([[Kf, None], [None, Kp]], format="csr") + Kif
        n_total = int(R.size)
        constrained = np.zeros(n_total, dtype=bool)
        rows = np.fromiter((int(k) for k in data_dirichlet.keys()), dtype=int)
        if rows.size:
            constrained[rows] = True
        r_free = float(np.linalg.norm(R[~constrained], ord=np.inf)) if np.any(~constrained) else 0.0
        print(f"\nCoupled residual (free dofs): |R|_inf = {r_free:.3e}")
        _print_block_norms(
            "Fluid volume residual (free dofs)",
            prob.dh_f,
            np.concatenate([Rf, np.zeros_like(Rp)]),
            ["v_pos_x", "v_pos_y", "p_pos_"],
            offset=0,
            constrained=constrained,
        )
        _print_block_norms(
            "Poro volume residual (free dofs)",
            prob.dh_p,
            np.concatenate([np.zeros_like(Rf), Rp]),
            ["v_neg_x", "v_neg_y", "u_neg_x", "u_neg_y", "p_neg_"],
            offset=int(prob.dh_f.total_dofs),
            constrained=constrained,
        )
        _print_block_norms(
            "Interface residual (free dofs)",
            prob.dh_f,
            Rif,
            ["v_pos_x", "v_pos_y", "p_pos_"],
            offset=0,
            constrained=constrained,
        )
        _print_block_norms(
            "Interface residual (free dofs)",
            prob.dh_p,
            Rif,
            ["v_neg_x", "v_neg_y", "u_neg_x", "u_neg_y", "p_neg_"],
            offset=int(prob.dh_f.total_dofs),
            constrained=constrained,
        )
        return

    for it in range(int(args.newton_it)):
        Uf = U[: int(prob.dh_f.total_dofs)]
        Up = U[int(prob.dh_f.total_dofs) :]

        # volume assemblies
        Kf, Rf = _assemble_fluid_volume(
            prob, Uf_k=Uf, Uf_n=Uf_n, dt=dt, rho_f=rho_f, mu_f=mu_f, qdeg=int(args.q), fF=fF
        )
        Kp, Rp = _assemble_poro_volume(
            prob,
            Up_k=Up,
            Up_n=Up_n,
            Up_nm1=Up_nm1,
            dt=dt,
            rho_f=rho_f,
            mu_f=mu_f,
            rho_s0=1.0,
            porosity=porosity,
            qdeg=int(args.q),
            K_inv=K_inv,
            c_nh=c_nh,
            beta_nh=beta_nh,
            f_mass=f_mass,
            fD=fD,
            fS=fS,
        )

        # interface coupling (python)
        Kif, Rif = assemble_fpi_interface_nitsche(
            interface=prob.iface_fp,
            dh_f=prob.dh_f,
            dh_p=prob.dh_p,
            Uf=Uf,
            Up=Up,
            Up_n=Up_n,
            dt=dt,
            rho_f=rho_f,
            mu_f=mu_f,
            porosity=porosity,
            beta_BJ=beta_BJ,
            kappa=kappa,
            gamma_n=gamma,
            gamma_t=gamma,
            zeta=zeta,
            vF_inf=v_inf,
            c_v_gamma=1.0 / 6.0,
            c_t_gamma=1.0 / 12.0,
            quad_order=int(args.q),
            g_sigma=g_sigma,
            g_sigma_n=g_sigma_n,
            g_n=g_n,
            g_t=g_t,
        )

        # inlet traction RHS on fluid (analytic)
        Rin = assemble_inlet_traction_rhs(
            inlet_P0=prob.inlet["inlet_P0"],
            inlet_P1=prob.inlet["inlet_P1"],
            inlet_pos_elem_ids=prob.inlet["inlet_pos_elem_ids"],
            dh_f=prob.dh_f,
            sigmaF_A=_sigmaF_A,
            quad_order=int(args.q),
            x0=float(args.x0),
        )

        # global K and residual (R = volume + interface + inlet traction)
        n_f = int(prob.dh_f.total_dofs)
        n_p = int(prob.dh_p.total_dofs)
        K = sp.bmat([[Kf, None], [None, Kp]], format="csr")
        R = np.concatenate([Rf + Rin, Rp])
        K = (K + Kif).tocsr()
        R = R + Rif

        # Apply strong constraints (Dirichlet + inactive) for the *increment* system.
        constrained = np.zeros(int(R.size), dtype=bool)
        rows = np.fromiter((int(k) for k in data_dirichlet.keys()), dtype=int)
        if rows.size:
            constrained[rows] = True
        res = float(np.linalg.norm(R[~constrained], ord=np.inf)) if np.any(~constrained) else 0.0
        print(f"Newton {it+1}: |R_free|_inf = {res:.3e}")
        if res < float(args.tol):
            break

        Kc, rhs = apply_dirichlet_increment(K, R, U, data_dirichlet)
        dU = spla.spsolve(Kc, rhs)
        U = U + np.asarray(dU, float)

    # Error summary (domain-wise, relative L2)
    err_vF = prob.dh_f.l2_error(
        U[: int(prob.dh_f.total_dofs)],
        exact={
            "v_pos_x": mms.vF_x,
            "v_pos_y": mms.vF_y,
        },
        quad_order=int(args.q),
        relative=False,
    )
    err_pF = prob.dh_f.l2_error(
        U[: int(prob.dh_f.total_dofs)],
        exact={"p_pos_": mms.pF_k},
        quad_order=int(args.q),
        relative=False,
    )
    err_vP = prob.dh_p.l2_error(
        U[int(prob.dh_f.total_dofs) :],
        exact={
            "v_neg_x": lambda x, y: float(np.asarray(mms.vP_k(x, y), float)[0]),
            "v_neg_y": lambda x, y: float(np.asarray(mms.vP_k(x, y), float)[1]),
        },
        quad_order=int(args.q),
        relative=False,
    )
    err_uP = prob.dh_p.l2_error(
        U[int(prob.dh_f.total_dofs) :],
        exact={
            "u_neg_x": lambda x, y: float(np.asarray(mms.uP_k(x, y), float)[0]),
            "u_neg_y": lambda x, y: float(np.asarray(mms.uP_k(x, y), float)[1]),
        },
        quad_order=int(args.q),
        relative=False,
    )
    err_pP = prob.dh_p.l2_error(
        U[int(prob.dh_f.total_dofs) :],
        exact={"p_neg_": mms.pP_k},
        quad_order=int(args.q),
        relative=False,
    )
    h = 1.0 / float(nx_f)
    print("\nTwo-mesh FPI Example 4.1 (nonmatching)")
    print(f"  nx_f={nx_f} nx_p={nx_p} h~{h:.3e}  |e(vF)|={err_vF:.3e}  |e(pF)|={err_pF:.3e}  |e(vP)|={err_vP:.3e}  |e(uP)|={err_uP:.3e}  |e(pP)|={err_pP:.3e}")

    if args.plot:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
        from pycutfem.io.visualization import plot_mesh_2

        fig, ax = plt.subplots(figsize=(10, 9))
        plot_mesh_2(prob.mesh_f, level_set=prob.fluid_ls, show=False, ax=ax, plot_nodes=False, plot_interface=True)
        ax.set_title("Fluid mesh (CutFEM) with fluid_ls=0")
        fig.savefig(outdir / f"mesh_fluid_nx{nx_f}_p{args.p}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 8))
        plot_mesh_2(prob.mesh_p, level_set=prob.poro_ls, show=False, ax=ax, plot_nodes=False, plot_interface=False)
        ax.set_title("Poro mesh (body-fitted) with poro_ls=0")
        fig.savefig(outdir / f"mesh_poro_nx{nx_p}_p{args.p}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
