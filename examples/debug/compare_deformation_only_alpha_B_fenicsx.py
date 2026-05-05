#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI
import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import ufl
from basix.ufl import mixed_element
from scipy.sparse import csr_matrix

from examples.utils.biofilm.deformation_only import build_deformation_only_forms
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant, Function, Identity, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction
from pycutfem.ufl.expressions import TestFunction as UflTestFunction
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def _one_to_one_map_coords(coords_pc: np.ndarray, coords_fx: np.ndarray, tol: float = 1.0e-12) -> np.ndarray:
    coords_pc = np.asarray(coords_pc, dtype=float)
    coords_fx = np.asarray(coords_fx, dtype=float)
    if coords_pc.shape != coords_fx.shape:
        raise ValueError(f"Coordinate shape mismatch: {coords_pc.shape} vs {coords_fx.shape}")
    scale = 1.0 / max(float(tol), 1.0e-16)

    def _quantize(arr: np.ndarray) -> np.ndarray:
        return np.rint(arr * scale).astype(np.int64)

    keys_pc = _quantize(coords_pc)
    keys_fx = _quantize(coords_fx)
    buckets: dict[tuple[int, ...], list[int]] = {}
    for i, key in enumerate(keys_fx):
        buckets.setdefault(tuple(int(v) for v in key.tolist()), []).append(int(i))

    mapping = np.empty(coords_pc.shape[0], dtype=int)
    for i, key in enumerate(keys_pc):
        bucket = buckets.get(tuple(int(v) for v in key.tolist()))
        if not bucket:
            raise KeyError(f"No coordinate match for {coords_pc[i].tolist()}")
        mapping[i] = int(bucket.pop())
    return mapping


def _compare_dense(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    denom = max(1.0, float(np.max(np.abs(a))) if np.size(a) else 0.0, float(np.max(np.abs(b))) if np.size(b) else 0.0)
    return max_abs, max_abs / denom


def _assemble_vector(*, problem, form, qdeg: int, backend: str) -> np.ndarray:
    _, residual = assemble_form(
        Equation(None, form),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=int(qdeg),
        backend=str(backend),
    )
    return np.asarray(residual, dtype=float)


def _assemble_matrix(*, problem, form, qdeg: int, backend: str) -> np.ndarray:
    matrix, _ = assemble_form(
        Equation(form, None),
        dof_handler=problem["dh"],
        bcs=[],
        quad_order=int(qdeg),
        backend=str(backend),
    )
    if hasattr(matrix, "to_scipy"):
        return matrix.to_scipy().toarray()
    return matrix.toarray()


def _build_pycutfem_problem(*, Lx: float, Ly: float, nx: int, ny: int, poly_order: int, scalar_order: int):
    nodes, elems, _, corners = structured_quad(float(Lx), float(Ly), nx=int(nx), ny=int(ny), poly_order=int(poly_order))
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=int(poly_order),
    )
    me = MixedElement(
        mesh,
        field_specs={
            "v_x": int(poly_order),
            "v_y": int(poly_order),
            "p": int(scalar_order),
            "vS_x": int(poly_order),
            "vS_y": int(poly_order),
            "u_x": int(poly_order),
            "u_y": int(poly_order),
            "alpha": int(scalar_order),
            "B": int(scalar_order),
            "mu_alpha": int(scalar_order),
        },
    )
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
    VS = FunctionSpace("VS", ["vS_x", "vS_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    problem = {
        "dh": dh,
        "dv": VectorTrialFunction(space=V, dof_handler=dh),
        "dp": TrialFunction("p", dof_handler=dh),
        "dvS": VectorTrialFunction(space=VS, dof_handler=dh),
        "du": VectorTrialFunction(space=U, dof_handler=dh),
        "dalpha": TrialFunction("alpha", dof_handler=dh),
        "dB": TrialFunction("B", dof_handler=dh),
        "dmu": TrialFunction("mu_alpha", dof_handler=dh),
        "v_test": VectorTestFunction(space=V, dof_handler=dh),
        "q_test": UflTestFunction("p", dof_handler=dh),
        "vS_test": VectorTestFunction(space=VS, dof_handler=dh),
        "u_test": VectorTestFunction(space=U, dof_handler=dh),
        "alpha_test": UflTestFunction("alpha", dof_handler=dh),
        "B_test": UflTestFunction("B", dof_handler=dh),
        "mu_test": UflTestFunction("mu_alpha", dof_handler=dh),
        "v_k": VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh),
        "p_k": Function("p_k", "p", dof_handler=dh),
        "vS_k": VectorFunction("vS_k", ["vS_x", "vS_y"], dof_handler=dh),
        "u_k": VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh),
        "alpha_k": Function("alpha_k", "alpha", dof_handler=dh),
        "B_k": Function("B_k", "B", dof_handler=dh),
        "mu_k": Function("mu_k", "mu_alpha", dof_handler=dh),
        "v_n": VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh),
        "p_n": Function("p_n", "p", dof_handler=dh),
        "vS_n": VectorFunction("vS_n", ["vS_x", "vS_y"], dof_handler=dh),
        "u_n": VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh),
        "alpha_n": Function("alpha_n", "alpha", dof_handler=dh),
        "B_n": Function("B_n", "B", dof_handler=dh),
        "mu_n": Function("mu_n", "mu_alpha", dof_handler=dh),
    }

    alpha_n_fun = lambda x, y: 0.55 + 0.12 * np.sin(np.pi * x) * np.cos(0.5 * np.pi * y / max(float(Ly), 1.0e-12))
    alpha_k_fun = lambda x, y: 0.60 + 0.10 * np.sin(0.5 * np.pi * x) * np.cos(np.pi * y / max(float(Ly), 1.0e-12))
    phi_n_fun = lambda x, y: 0.35 + 0.08 * x
    phi_k_fun = lambda x, y: 0.45 + 0.05 * y / max(float(Ly), 1.0e-12)

    problem["v_n"].set_values_from_function(lambda x, y: np.array([0.03 + 0.01 * x, -0.02 + 0.015 * y]))
    problem["v_k"].set_values_from_function(lambda x, y: np.array([0.05 + 0.015 * x, -0.03 + 0.020 * y]))
    problem["p_n"].set_values_from_function(lambda x, y: 0.15 + 0.05 * x - 0.03 * y)
    problem["p_k"].set_values_from_function(lambda x, y: 0.22 + 0.07 * x - 0.04 * y)
    problem["vS_n"].set_values_from_function(lambda x, y: np.array([0.01 + 0.01 * x, -0.01 + 0.005 * y]))
    problem["vS_k"].set_values_from_function(lambda x, y: np.array([0.02 + 0.012 * x, -0.015 + 0.008 * y]))
    problem["u_n"].set_values_from_function(lambda x, y: np.array([0.01 * x * (1.0 - x), -0.004 * y]))
    problem["u_k"].set_values_from_function(lambda x, y: np.array([0.015 * x * (1.0 - x), -0.006 * y]))
    problem["alpha_n"].set_values_from_function(alpha_n_fun)
    problem["alpha_k"].set_values_from_function(alpha_k_fun)
    problem["B_n"].set_values_from_function(lambda x, y: alpha_n_fun(x, y) * (1.0 - phi_n_fun(x, y)))
    problem["B_k"].set_values_from_function(lambda x, y: alpha_k_fun(x, y) * (1.0 - phi_k_fun(x, y)))
    problem["mu_n"].set_values_from_function(lambda x, y: 0.05 * alpha_n_fun(x, y))
    problem["mu_k"].set_values_from_function(lambda x, y: 0.07 * alpha_k_fun(x, y) + 0.01 * y)
    return problem


def _build_pycutfem_forms(problem, *, qdeg: int):
    return build_deformation_only_forms(
        v_k=problem["v_k"],
        p_k=problem["p_k"],
        vS_k=problem["vS_k"],
        u_k=problem["u_k"],
        alpha_k=problem["alpha_k"],
        B_k=problem["B_k"],
        mu_alpha_k=problem["mu_k"],
        v_n=problem["v_n"],
        p_n=problem["p_n"],
        vS_n=problem["vS_n"],
        u_n=problem["u_n"],
        alpha_n=problem["alpha_n"],
        B_n=problem["B_n"],
        mu_alpha_n=problem["mu_n"],
        dv=problem["dv"],
        dp=problem["dp"],
        dvS=problem["dvS"],
        du=problem["du"],
        dalpha=problem["dalpha"],
        dB=problem["dB"],
        dmu_alpha=problem["dmu"],
        v_test=problem["v_test"],
        q_test=problem["q_test"],
        vS_test=problem["vS_test"],
        u_test=problem["u_test"],
        alpha_test=problem["alpha_test"],
        B_test=problem["B_test"],
        mu_alpha_test=problem["mu_test"],
        dx=dx(metadata={"q": int(qdeg)}),
        dt=Constant(0.1),
        theta=1.0,
        rho_f=Constant(1.0),
        mu_f=Constant(0.035),
        mu_b=Constant(0.035),
        kappa_inv=Constant(1.0e5),
        mu_s=Constant(1.67785e5),
        lambda_s=Constant(8.22148e6),
        solid_model="linear",
        kappa_inv_model="refmap",
        M_alpha=1.0,
        gamma_alpha=1.0,
        eps_alpha=0.1,
        support_physics="internal_conversion",
        alpha_advect_with="biofilm_volume",
        alpha_advection_form="conservative_weak",
        fluid_convection="full",
        include_skeleton_acceleration=True,
        rho_s0_tilde=Constant(1.1),
        skeleton_inertia_convection="full",
        skeleton_pressure_mode="whole_domain",
    )


def _build_fenics_space(*, Lx: float, Ly: float, nx: int, ny: int, poly_order: int, scalar_order: int):
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0], dtype=float), np.array([float(Lx), float(Ly)], dtype=float)],
        [int(nx), int(ny)],
        cell_type=dolfinx.mesh.CellType.quadrilateral,
    )
    gdim = mesh.geometry.dim
    V_el = basix.ufl.element("Lagrange", "quadrilateral", int(poly_order), shape=(gdim,))
    A_el = basix.ufl.element("Lagrange", "quadrilateral", int(scalar_order))
    W_el = mixed_element([V_el, A_el, V_el, V_el, A_el, A_el, A_el])
    if hasattr(dolfinx.fem, "functionspace"):
        W = dolfinx.fem.functionspace(mesh, W_el)
    else:
        W = dolfinx.fem.FunctionSpace(mesh, W_el)
    return mesh, W


def _fenics_parent_dofs_for_component(W, subspace_index: int, component: int) -> tuple[np.ndarray, np.ndarray]:
    Wv, V_map = W.sub(subspace_index).collapse()
    Wc, C_map = Wv.sub(component).collapse()
    coords = Wc.tabulate_dof_coordinates()[:, :2]
    parent = np.asarray(V_map, dtype=int)[np.asarray(C_map, dtype=int)]
    return parent, coords


def _fenics_parent_dofs_for_scalar(W, subspace_index: int) -> tuple[np.ndarray, np.ndarray]:
    Ws, S_map = W.sub(subspace_index).collapse()
    coords = Ws.tabulate_dof_coordinates()[:, :2]
    parent = np.asarray(S_map, dtype=int)
    return parent, coords


def _build_mapping(problem, W) -> dict[str, np.ndarray]:
    dh = problem["dh"]
    mapping: dict[str, np.ndarray] = {}
    vector_fields = (
        ("v_x", 0, 0),
        ("v_y", 0, 1),
        ("vS_x", 2, 0),
        ("vS_y", 2, 1),
        ("u_x", 3, 0),
        ("u_y", 3, 1),
    )
    scalar_fields = (("p", 1), ("alpha", 4), ("B", 5), ("mu_alpha", 6))
    for fld, sub_idx, comp in vector_fields:
        parent, coords_fx = _fenics_parent_dofs_for_component(W, sub_idx, comp)
        coords_pc = np.asarray(problem["dh"].get_dof_coords(fld), dtype=float)
        mapping[fld] = parent[_one_to_one_map_coords(coords_pc, coords_fx)]
    for fld, sub_idx in scalar_fields:
        parent, coords_fx = _fenics_parent_dofs_for_scalar(W, sub_idx)
        coords_pc = np.asarray(dh.get_dof_coords(fld), dtype=float)
        mapping[fld] = parent[_one_to_one_map_coords(coords_pc, coords_fx)]
    return mapping


def _load_fenics_state(problem, W, mapping: dict[str, np.ndarray]):
    w_k = dolfinx.fem.Function(W, name="w_k")
    w_n = dolfinx.fem.Function(W, name="w_n")
    fields_k = [
        ("v_x", problem["v_k"].components[0]),
        ("v_y", problem["v_k"].components[1]),
        ("p", problem["p_k"]),
        ("vS_x", problem["vS_k"].components[0]),
        ("vS_y", problem["vS_k"].components[1]),
        ("u_x", problem["u_k"].components[0]),
        ("u_y", problem["u_k"].components[1]),
        ("alpha", problem["alpha_k"]),
        ("B", problem["B_k"]),
        ("mu_alpha", problem["mu_k"]),
    ]
    fields_n = [
        ("v_x", problem["v_n"].components[0]),
        ("v_y", problem["v_n"].components[1]),
        ("p", problem["p_n"]),
        ("vS_x", problem["vS_n"].components[0]),
        ("vS_y", problem["vS_n"].components[1]),
        ("u_x", problem["u_n"].components[0]),
        ("u_y", problem["u_n"].components[1]),
        ("alpha", problem["alpha_n"]),
        ("B", problem["B_n"]),
        ("mu_alpha", problem["mu_n"]),
    ]
    for fld, func in fields_k:
        w_k.x.array[mapping[fld]] = np.asarray(func.nodal_values, dtype=float)
    for fld, func in fields_n:
        w_n.x.array[mapping[fld]] = np.asarray(func.nodal_values, dtype=float)
    w_k.x.scatter_forward()
    w_n.x.scatter_forward()
    return w_k, w_n


def _fenics_forms(W, w_k, w_n, *, qdeg: int):
    mesh = W.mesh
    gdim = mesh.geometry.dim
    dw = ufl.TrialFunction(W)
    w_test = ufl.TestFunction(W)
    v_test, q_test, vS_test, u_test, alpha_test, B_test, mu_test = ufl.split(w_test)
    v_k, p_k, vS_k, u_k, alpha_k, B_k, mu_k = ufl.split(w_k)
    v_n, p_n, vS_n, u_n, alpha_n, B_n, mu_n = ufl.split(w_n)
    I = ufl.Identity(gdim)

    def _const(val: float):
        return dolfinx.fem.Constant(mesh, float(val))

    def _eps(v):
        return ufl.sym(ufl.grad(v))

    dt = _const(0.1)
    theta = _const(1.0)
    one_m_theta = _const(0.0)
    inv_dt = _const(10.0)
    rho_f = _const(1.0)
    mu_f = _const(0.035)
    mu_b = _const(0.035)
    kappa_inv = _const(1.0e5)
    mu_s = _const(1.67785e5)
    lambda_s = _const(8.22148e6)
    rho_s0 = _const(1.1)
    M_alpha = _const(1.0)
    gamma_alpha = _const(1.0)
    eps_alpha = _const(0.1)

    P_k = alpha_k - B_k
    P_n = alpha_n - B_n
    C_k = 1.0 - B_k
    C_n = 1.0 - B_n
    phi_sq_k = (P_k * P_k) / (alpha_k * alpha_k + 1.0e-12)
    phi_sq_n = (P_n * P_n) / (alpha_n * alpha_n + 1.0e-12)
    rho_k = rho_f * C_k
    rho_n = rho_f * C_n
    mu_mix_k = (1.0 - alpha_k) * mu_f + alpha_k * mu_b
    mu_mix_n = (1.0 - alpha_n) * mu_f + alpha_n * mu_b

    gradC_k = -ufl.grad(B_k)
    gradC_n = -ufl.grad(B_n)
    gradB_k = ufl.grad(B_k)
    gradB_n = ufl.grad(B_n)
    gradP_k = ufl.grad(alpha_k) - ufl.grad(B_k)
    gradP_n = ufl.grad(alpha_n) - ufl.grad(B_n)

    div_C_vtest = C_k * ufl.div(v_test) + ufl.dot(gradC_k, v_test)
    div_B_vStest_k = B_k * ufl.div(vS_test) + ufl.dot(gradB_k, vS_test)
    div_B_vStest_n = B_n * ufl.div(vS_test) + ufl.dot(gradB_n, vS_test)
    div_C_vk = C_k * ufl.div(v_k) + ufl.dot(gradC_k, v_k)
    div_C_vn = C_n * ufl.div(v_n) + ufl.dot(gradC_n, v_n)
    div_BvS_k = B_k * ufl.div(vS_k) + ufl.dot(gradB_k, vS_k)
    div_BvS_n = B_n * ufl.div(vS_n) + ufl.dot(gradB_n, vS_n)
    div_rhov_k = rho_f * div_C_vk
    div_rhov_n = rho_f * div_C_vn

    dx_fx = ufl.dx(metadata={"quadrature_degree": int(qdeg)})

    r_mom = ufl.inner((rho_k * v_k - rho_n * v_n) * inv_dt, v_test) * dx_fx
    conv_k = ufl.dot(ufl.dot(ufl.grad(v_k), v_k), v_test)
    conv_n = ufl.dot(ufl.dot(ufl.grad(v_n), v_n), v_test)
    r_mom += (theta * (rho_k * conv_k + div_rhov_k * ufl.dot(v_k, v_test)) + one_m_theta * (rho_n * conv_n + div_rhov_n * ufl.dot(v_n, v_test))) * dx_fx
    r_mom += 2.0 * (theta * mu_mix_k * ufl.inner(_eps(v_k), _eps(v_test)) + one_m_theta * mu_mix_n * ufl.inner(_eps(v_n), _eps(v_test))) * dx_fx
    r_mom += -(p_k * div_C_vtest) * dx_fx

    K_inv_ref = kappa_inv * I
    F_inv_k = I - ufl.grad(u_k)
    F_inv_n = I - ufl.grad(u_n)
    F_k = ufl.inv(F_inv_k)
    F_n = ufl.inv(F_inv_n)
    J_k = ufl.det(F_k)
    J_n = ufl.det(F_n)
    k_inv_k = J_k * ufl.dot(F_inv_k.T, ufl.dot(K_inv_ref, F_inv_k))
    k_inv_n = J_n * ufl.dot(F_inv_n.T, ufl.dot(K_inv_ref, F_inv_n))
    beta_coeff_k = B_k * mu_f * phi_sq_k
    beta_coeff_n = B_n * mu_f * phi_sq_n
    r_mom += beta_coeff_k * ufl.dot(ufl.dot(k_inv_k, v_k - vS_k), v_test) * dx_fx

    r_mass = q_test * (div_C_vk + div_BvS_k) * dx_fx

    r_el_k = 2.0 * mu_s * ufl.inner(_eps(u_k), _eps(vS_test)) + lambda_s * ufl.div(u_k) * ufl.div(vS_test)
    r_el_n = 2.0 * mu_s * ufl.inner(_eps(u_n), _eps(vS_test)) + lambda_s * ufl.div(u_n) * ufl.div(vS_test)
    r_skel_press_k = -(p_k * div_B_vStest_k)
    r_skel_press_n = -(p_n * div_B_vStest_n)
    r_skel_drag_k = -beta_coeff_k * ufl.dot(ufl.dot(k_inv_k, v_k - vS_k), vS_test)
    r_skel_drag_n = -beta_coeff_n * ufl.dot(ufl.dot(k_inv_n, v_n - vS_n), vS_test)
    r_skel = (theta * alpha_k * r_el_k + one_m_theta * alpha_n * r_el_n + theta * r_skel_press_k + one_m_theta * r_skel_press_n + theta * r_skel_drag_k + one_m_theta * r_skel_drag_n) * dx_fx

    rhoS_k = rho_s0 * B_k
    rhoS_n = rho_s0 * B_n
    r_skel += ufl.inner((rhoS_k * vS_k - rhoS_n * vS_n) * inv_dt, vS_test) * dx_fx
    advS_k = ufl.dot(ufl.dot(ufl.grad(vS_k), vS_k), vS_test)
    advS_n = ufl.dot(ufl.dot(ufl.grad(vS_n), vS_n), vS_test)
    r_skel += theta * (rhoS_k * advS_k + rho_s0 * div_BvS_k * ufl.dot(vS_k, vS_test)) * dx_fx
    r_skel += one_m_theta * (rhoS_n * advS_n + rho_s0 * div_BvS_n * ufl.dot(vS_n, vS_test)) * dx_fx

    Fkin_k = (u_k - u_n) * inv_dt + theta * (ufl.dot(ufl.grad(u_k), vS_k) - vS_k) + one_m_theta * (ufl.dot(ufl.grad(u_n), vS_n) - vS_n)
    r_kin = rho_s0 * alpha_k * ufl.dot(Fkin_k, u_test) * dx_fx

    q_alpha_k = P_k * v_k + B_k * vS_k
    q_alpha_n = P_n * v_n + B_n * vS_n
    r_alpha = alpha_test * ((alpha_k - alpha_n) * inv_dt) * dx_fx
    r_alpha += -theta * ufl.dot(q_alpha_k, ufl.grad(alpha_test)) * dx_fx
    r_alpha += -one_m_theta * ufl.dot(q_alpha_n, ufl.grad(alpha_test)) * dx_fx
    r_alpha += M_alpha * ufl.inner(ufl.grad(mu_k), ufl.grad(alpha_test)) * dx_fx

    r_B = B_test * ((B_k - B_n) * inv_dt) * dx_fx
    r_B += -theta * ufl.dot(B_k * vS_k, ufl.grad(B_test)) * dx_fx
    r_B += -one_m_theta * ufl.dot(B_n * vS_n, ufl.grad(B_test)) * dx_fx

    Wp = 2.0 * alpha_k * (1.0 - alpha_k) * (1.0 - 2.0 * alpha_k)
    r_mu = mu_test * mu_k * dx_fx
    r_mu += -(gamma_alpha * eps_alpha) * ufl.inner(ufl.grad(alpha_k), ufl.grad(mu_test)) * dx_fx
    r_mu += -mu_test * ((gamma_alpha / eps_alpha) * Wp) * dx_fx

    r_total = r_mom + r_mass + r_skel + r_kin + r_alpha + r_B + r_mu
    a_total = ufl.derivative(r_total, w_k, dw)
    return r_total, a_total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=1)
    ap.add_argument("--ny", type=int, default=1)
    ap.add_argument("--poly-order", type=int, default=1)
    ap.add_argument("--scalar-order", type=int, default=1)
    ap.add_argument("--qdeg", type=int, default=4)
    ap.add_argument("--pycutfem-backend", type=str, default="python", choices=("python", "jit", "cpp"))
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    Lx = 1.0
    Ly = 0.25
    qdeg = int(args.qdeg)

    problem = _build_pycutfem_problem(
        Lx=Lx,
        Ly=Ly,
        nx=int(args.nx),
        ny=int(args.ny),
        poly_order=int(args.poly_order),
        scalar_order=int(args.scalar_order),
    )
    forms = _build_pycutfem_forms(problem, qdeg=qdeg)
    R_pc = _assemble_vector(problem=problem, form=forms.residual_form, qdeg=qdeg, backend=str(args.pycutfem_backend))
    A_pc = _assemble_matrix(problem=problem, form=forms.jacobian_form, qdeg=qdeg, backend=str(args.pycutfem_backend))

    mesh_fx, W_fx = _build_fenics_space(
        Lx=Lx,
        Ly=Ly,
        nx=int(args.nx),
        ny=int(args.ny),
        poly_order=int(args.poly_order),
        scalar_order=int(args.scalar_order),
    )
    mapping = _build_mapping(problem, W_fx)
    w_k_fx, w_n_fx = _load_fenics_state(problem, W_fx, mapping)
    r_fx, a_fx = _fenics_forms(W_fx, w_k_fx, w_n_fx, qdeg=qdeg)
    vec_fx = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(r_fx))
    vec_fx.assemble()
    A_fx = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a_fx))
    A_fx.assemble()

    perm = np.empty((int(problem["dh"].total_dofs),), dtype=int)
    for fld in ("v_x", "v_y", "p", "vS_x", "vS_y", "u_x", "u_y", "alpha", "B", "mu_alpha"):
        sl = np.asarray(problem["dh"].get_field_slice(fld), dtype=int)
        perm[sl] = np.asarray(mapping[fld], dtype=int)

    R_fx_reordered = np.asarray(vec_fx.array, dtype=float)[perm]
    Ai, Aj, Av = A_fx.getValuesCSR()
    A_fx_dense = csr_matrix((Av, Aj, Ai), shape=A_fx.size).toarray()
    A_fx_reordered = A_fx_dense[np.ix_(perm, perm)]

    full_res_max_abs, full_res_rel = _compare_dense(R_pc, R_fx_reordered)
    full_jac_max_abs, full_jac_rel = _compare_dense(A_pc, A_fx_reordered)

    row_blocks: dict[str, dict[str, float]] = {}
    for fld in ("v_x", "v_y", "p", "vS_x", "vS_y", "u_x", "u_y", "alpha", "B", "mu_alpha"):
        sl = np.asarray(problem["dh"].get_field_slice(fld), dtype=int)
        res_abs, res_rel = _compare_dense(R_pc[sl], R_fx_reordered[sl])
        jac_abs, jac_rel = _compare_dense(A_pc[np.ix_(sl, np.arange(A_pc.shape[1]))], A_fx_reordered[np.ix_(sl, np.arange(A_pc.shape[1]))])
        row_blocks[str(fld)] = {
            "res_max_abs": float(res_abs),
            "res_rel": float(res_rel),
            "jac_max_abs": float(jac_abs),
            "jac_rel": float(jac_rel),
        }

    result = {
        "pycutfem_backend": str(args.pycutfem_backend),
        "full_res_max_abs": float(full_res_max_abs),
        "full_res_rel": float(full_res_rel),
        "full_jac_max_abs": float(full_jac_max_abs),
        "full_jac_rel": float(full_jac_rel),
        "row_blocks": row_blocks,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
