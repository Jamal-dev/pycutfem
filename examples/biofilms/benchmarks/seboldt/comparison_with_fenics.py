#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mpi4py import MPI
import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import ufl
from basix.ufl import mixed_element
from scipy.sparse import csr_matrix

from pycutfem.ufl.expressions import Constant
from pycutfem.ufl.forms import Equation, assemble_form
from examples.biofilms.benchmarks.seboldt.paper1_benchmark7_seboldt import (
    _build_forms,
    _create_problem,
)


@dataclass
class CaseSummary:
    name: str
    directional_fd_max_rel: float
    directional_fd_per_case: dict[str, float]
    alpha_res_max_abs: float
    alpha_res_rel: float
    alpha_jac_max_abs: float
    alpha_jac_rel: float


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


def _resolved_orders(poly_order: int, pressure_order: int | None, scalar_order: int | None) -> tuple[int, int, int]:
    p = int(poly_order)
    pp = int(max(1, p - 1) if pressure_order is None else pressure_order)
    ps = int(max(1, p - 1) if scalar_order is None else scalar_order)
    return p, pp, ps


def _initialize_problem(problem: dict[str, object], *, enable_phi_evolution: bool) -> None:
    problem["v_k"].set_values_from_function(lambda x, y: np.array([0.18 + 0.04 * x - 0.03 * y, -0.06 + 0.02 * x + 0.015 * y]))
    problem["v_n"].set_values_from_function(lambda x, y: np.array([0.11 + 0.03 * x - 0.02 * y, -0.03 + 0.015 * x + 0.010 * y]))
    problem["vS_k"].set_values_from_function(lambda x, y: np.array([0.05 + 0.02 * x - 0.01 * y, -0.03 + 0.01 * x + 0.015 * y]))
    problem["vS_n"].set_values_from_function(lambda x, y: np.array([0.03 + 0.01 * x - 0.008 * y, -0.015 + 0.008 * x + 0.010 * y]))
    problem["u_k"].set_values_from_function(lambda x, y: np.array([0.015 + 0.010 * x * (1.0 - x), -0.008 + 0.006 * y * (1.0 - y / 1.5)]))
    problem["u_n"].set_values_from_function(lambda x, y: np.array([0.010 + 0.006 * x * (1.0 - x), -0.005 + 0.004 * y * (1.0 - y / 1.5)]))
    problem["p_k"].set_values_from_function(lambda x, y: 0.25 + 0.10 * x - 0.05 * y)
    problem["p_n"].set_values_from_function(lambda x, y: 0.12 + 0.04 * x - 0.03 * y)
    problem["alpha_k"].set_values_from_function(lambda x, y: 0.58 + 0.015 * x + 0.010 * y)
    problem["alpha_n"].set_values_from_function(lambda x, y: 0.54 + 0.010 * x + 0.008 * y)
    problem["mu_k"].set_values_from_function(lambda x, y: 0.10 + 0.020 * x - 0.015 * y)
    problem["mu_n"].set_values_from_function(lambda x, y: 0.07 + 0.015 * x - 0.010 * y)
    if bool(enable_phi_evolution):
        problem["phi_k"].set_values_from_function(lambda x, y: 0.68 + 0.04 * x - 0.03 * y)
        problem["phi_n"].set_values_from_function(lambda x, y: 0.66 + 0.03 * x - 0.02 * y)
        problem["S_k"].set_values_from_function(lambda x, y: 0.20 + 0.02 * x + 0.01 * y)
        problem["S_n"].set_values_from_function(lambda x, y: 0.17 + 0.015 * x + 0.008 * y)


def _field_to_func(problem: dict[str, object], *, enable_phi_evolution: bool) -> dict[str, object]:
    out = {
        "v_x": problem["v_k"].components[0],
        "v_y": problem["v_k"].components[1],
        "p": problem["p_k"],
        "vS_x": problem["vS_k"].components[0],
        "vS_y": problem["vS_k"].components[1],
        "u_x": problem["u_k"].components[0],
        "u_y": problem["u_k"].components[1],
        "alpha": problem["alpha_k"],
        "mu_alpha": problem["mu_k"],
    }
    if bool(enable_phi_evolution):
        out["phi"] = problem["phi_k"]
        out["S"] = problem["S_k"]
    return out


def _build_benchmark_case(
    *,
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    poly_order: int,
    pressure_order: int | None,
    scalar_order: int | None,
    dt: float,
    theta: float,
    enable_phi_evolution: bool,
    alpha_advect_with: str,
    alpha_advection_form: str,
    outdir: Path | None = None,
) -> tuple[dict[str, object], object, dict[str, object], int]:
    poly_order, pressure_order, scalar_order = _resolved_orders(poly_order, pressure_order, scalar_order)
    qdeg = max(6, 2 * int(poly_order) + 2)
    problem = _create_problem(
        Lx=float(Lx),
        Ly=float(Ly),
        nx=int(nx),
        ny=int(ny),
        poly_order=int(poly_order),
        pressure_order=int(pressure_order),
        scalar_order=int(scalar_order),
        fluid_space="cg",
        fluid_hdiv_order=0,
        enable_phi_evolution=bool(enable_phi_evolution),
    )
    _initialize_problem(problem, enable_phi_evolution=bool(enable_phi_evolution))
    forms = _build_forms(
        problem,
        qdeg=int(qdeg),
        dt_c=Constant(float(dt)),
        theta=float(theta),
        rho_f=1.0,
        mu_f=0.035,
        mu_b=0.035,
        mu_b_model="mu",
        kappa_inv=1.0e5,
        mu_s=1.67785e5,
        lambda_s=8.22148e6,
        phi_b=0.18,
        M_alpha=1.0,
        gamma_alpha=1.0,
        eps_alpha=0.05,
        solid_visco_eta=0.0,
        gamma_div=0.0,
        gamma_u=1.0,
        u_extension_mode="grad",
        gamma_u_pin=1.0e-6,
        u_cip=0.0,
        u_cip_weight="fluid",
        vS_cip=1.0,
        gamma_vS=1.0,
        vS_extension_mode="grad",
        gamma_vS_pin=1.0e-6,
        D_phi=0.0,
        phi_diffusion_weight="fluid",
        gamma_phi=5.0,
        phi_supg=0.0,
        phi_cip=1.0,
        alpha_regularization="none",
        alpha_reg_gamma=1.0,
        alpha_reg_eps_normal=0.05,
        alpha_reg_eps_tangent=0.0125,
        alpha_reg_eta=1.0e-12,
        alpha_advect_with=str(alpha_advect_with),
        alpha_advection_form=str(alpha_advection_form),
        solid_model="linear",
        kappa_inv_model="refmap",
        enable_phi_evolution=bool(enable_phi_evolution),
    )
    problem["_audit_Lx"] = float(Lx)
    problem["_audit_Ly"] = float(Ly)
    problem["_audit_nx"] = int(nx)
    problem["_audit_ny"] = int(ny)
    problem["_audit_poly_order"] = int(poly_order)
    problem["_audit_scalar_order"] = int(scalar_order)
    problem["_audit_dt"] = float(dt)
    problem["_audit_theta"] = float(theta)
    return problem, forms, _field_to_func(problem, enable_phi_evolution=bool(enable_phi_evolution)), int(qdeg)


def _assemble_residual(forms, dh, qdeg: int, *, backend: str) -> np.ndarray:
    _, r = assemble_form(Equation(None, forms.residual_form), dof_handler=dh, bcs=[], quad_order=int(qdeg), backend=str(backend))
    return np.asarray(r, dtype=float)


def _directional_fd_audit(
    problem,
    forms,
    field_to_func: dict[str, object],
    *,
    qdeg: int,
    enable_phi_evolution: bool,
    backend: str,
    n_random: int = 1,
    seed: int = 1234,
) -> tuple[float, dict[str, float]]:
    dh = problem["dh"]
    K, R0 = assemble_form(
        Equation(forms.jacobian_form, forms.residual_form),
        dof_handler=dh,
        bcs=[],
        quad_order=int(qdeg),
        backend=str(backend),
    )
    K = K.tocsr()
    R0 = np.asarray(R0, dtype=float)
    ndofs = int(K.shape[1])
    probe_fields = ["v_x", "v_y", "p", "vS_x", "vS_y", "u_x", "u_y", "alpha", "mu_alpha"]
    if bool(enable_phi_evolution):
        probe_fields.extend(["phi", "S"])

    rng = np.random.default_rng(int(seed))
    directions: list[tuple[str, np.ndarray]] = []

    def _field_direction(name: str, fields: list[str]) -> None:
        z = np.zeros(ndofs, dtype=float)
        for fld in fields:
            sl = np.asarray(dh.get_field_slice(fld), dtype=int)
            if sl.size:
                z[sl] = rng.standard_normal(sl.size)
        scale = float(np.linalg.norm(z, ord=np.inf))
        if scale > 0.0:
            directions.append((name, z / scale))

    _field_direction("alpha_only", ["alpha"])
    _field_direction("v_only", ["v_x", "v_y"])
    _field_direction("vS_only", ["vS_x", "vS_y"])
    _field_direction("u_only", ["u_x", "u_y"])
    if bool(enable_phi_evolution):
        _field_direction("phiS_only", ["phi", "S"])
    _field_direction("random_all", probe_fields)
    for i in range(int(n_random) - 1):
        _field_direction(f"random_all_{i + 2}", probe_fields)

    eps = 1.0e-8
    per_case: dict[str, float] = {}
    max_rel = 0.0
    for name, z in directions:
        touched: list[tuple[object, np.ndarray, np.ndarray]] = []
        for fld, func in field_to_func.items():
            sl = np.asarray(dh.get_field_slice(fld), dtype=int)
            if sl.size == 0:
                continue
            dz = z[sl]
            if np.allclose(dz, 0.0):
                continue
            old = np.asarray(func.get_nodal_values(sl), dtype=float).copy()
            func.set_nodal_values(sl, old + eps * dz)
            touched.append((func, sl, old))
        R1 = _assemble_residual(forms, dh, int(qdeg), backend=str(backend))
        for func, sl, old in touched:
            func.set_nodal_values(sl, old)

        fd = (R1 - R0) / eps
        lin = K @ z
        denom = max(1.0, float(np.linalg.norm(fd, ord=np.inf)), float(np.linalg.norm(lin, ord=np.inf)))
        rel = float(np.linalg.norm(fd - lin, ord=np.inf)) / denom
        per_case[name] = rel
        max_rel = max(max_rel, rel)
    return max_rel, per_case


def _build_fenics_alpha_system(*, Lx: float, Ly: float, nx: int, ny: int, poly_order: int, scalar_order: int):
    mesh_fx = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0], dtype=float), np.array([float(Lx), float(Ly)], dtype=float)],
        [int(nx), int(ny)],
        cell_type=dolfinx.mesh.CellType.quadrilateral,
    )
    gdim = mesh_fx.geometry.dim
    V_el = basix.ufl.element("Lagrange", "quadrilateral", int(poly_order), shape=(gdim,))
    A_el = basix.ufl.element("Lagrange", "quadrilateral", int(scalar_order))
    W_el = mixed_element([V_el, V_el, A_el])
    if hasattr(dolfinx.fem, "functionspace"):
        W = dolfinx.fem.functionspace(mesh_fx, W_el)
    else:
        W = dolfinx.fem.FunctionSpace(mesh_fx, W_el)
    return mesh_fx, W


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


def _map_pycutfem_to_fenics_alpha(problem, W) -> dict[str, np.ndarray]:
    dh = problem["dh"]
    mapping: dict[str, np.ndarray] = {}
    for fld, sub_idx, comp in (("v_x", 0, 0), ("v_y", 0, 1), ("vS_x", 1, 0), ("vS_y", 1, 1)):
        parent, coords_fx = _fenics_parent_dofs_for_component(W, sub_idx, comp)
        coords_pc = np.asarray(dh.get_dof_coords(fld), dtype=float)
        mapping[fld] = parent[_one_to_one_map_coords(coords_pc, coords_fx)]
    parent_alpha, coords_alpha = _fenics_parent_dofs_for_scalar(W, 2)
    mapping["alpha"] = parent_alpha[_one_to_one_map_coords(np.asarray(dh.get_dof_coords("alpha"), dtype=float), coords_alpha)]
    return mapping


def _load_fenics_state(problem, W, mapping: dict[str, np.ndarray]) -> tuple[object, object]:
    w_k = dolfinx.fem.Function(W, name="w_k")
    w_n = dolfinx.fem.Function(W, name="w_n")
    for fld, func in (
        ("v_x", problem["v_k"].components[0]),
        ("v_y", problem["v_k"].components[1]),
        ("vS_x", problem["vS_k"].components[0]),
        ("vS_y", problem["vS_k"].components[1]),
        ("alpha", problem["alpha_k"]),
    ):
        w_k.x.array[mapping[fld]] = np.asarray(func.nodal_values, dtype=float)
    for fld, func in (
        ("v_x", problem["v_n"].components[0]),
        ("v_y", problem["v_n"].components[1]),
        ("vS_x", problem["vS_n"].components[0]),
        ("vS_y", problem["vS_n"].components[1]),
        ("alpha", problem["alpha_n"]),
    ):
        w_n.x.array[mapping[fld]] = np.asarray(func.nodal_values, dtype=float)
    w_k.x.scatter_forward()
    w_n.x.scatter_forward()
    return w_k, w_n


def _fenics_alpha_forms(*, W, w_k, w_n, dt: float, theta: float, qdeg: int, alpha_advect_with: str, alpha_advection_form: str):
    dw = ufl.TrialFunction(W)
    wtest = ufl.TestFunction(W)
    dv_fx, dvS_fx, dalpha_fx = ufl.split(dw)
    zeta_v_fx, zeta_vS_fx, xi_fx = ufl.split(wtest)
    v_k_fx, vS_k_fx, alpha_k_fx = ufl.split(w_k)
    v_n_fx, vS_n_fx, alpha_n_fx = ufl.split(w_n)

    th_fx = dolfinx.fem.Constant(W.mesh, float(theta))
    omth_fx = dolfinx.fem.Constant(W.mesh, 1.0 - float(theta))
    inv_dt_fx = dolfinx.fem.Constant(W.mesh, 1.0 / float(dt))
    dx_fx = ufl.dx(metadata={"quadrature_degree": int(qdeg)})

    adv_key = str(alpha_advect_with).strip().lower()
    if adv_key == "vs":
        adv_u_k, adv_u_n = vS_k_fx, vS_n_fx
    elif adv_key == "v":
        adv_u_k, adv_u_n = v_k_fx, v_n_fx
    elif adv_key == "interface":
        adv_u_k = 0.5 * (v_k_fx + vS_k_fx)
        adv_u_n = 0.5 * (v_n_fx + vS_n_fx)
    else:
        raise ValueError(f"Unsupported alpha_advect_with={alpha_advect_with!r}")

    form_key = str(alpha_advection_form).strip().lower()
    if form_key == "conservative":
        a_k = alpha_k_fx
        a_n = alpha_n_fx
    elif form_key == "interface_band_conservative":
        a_k = 4.0 * alpha_k_fx * (1.0 - alpha_k_fx)
        a_n = 4.0 * alpha_n_fx * (1.0 - alpha_n_fx)
    else:
        raise ValueError(f"Unsupported alpha_advection_form={alpha_advection_form!r}")

    r_alpha_fx = xi_fx * ((a_k - a_n) * inv_dt_fx) * dx_fx
    r_alpha_fx += th_fx * xi_fx * (ufl.dot(ufl.grad(a_k), adv_u_k) + a_k * ufl.div(adv_u_k)) * dx_fx
    r_alpha_fx += omth_fx * xi_fx * (ufl.dot(ufl.grad(a_n), adv_u_n) + a_n * ufl.div(adv_u_n)) * dx_fx
    a_alpha_fx = ufl.derivative(r_alpha_fx, w_k, dw)
    return r_alpha_fx, a_alpha_fx


def _pycutfem_alpha_compare(
    problem,
    forms,
    *,
    qdeg: int,
    alpha_advect_with: str,
    alpha_advection_form: str,
    backend: str,
) -> tuple[float, float, float, float]:
    mesh_fx, W_fx = _build_fenics_alpha_system(
        Lx=float(problem["_audit_Lx"]),
        Ly=float(problem["_audit_Ly"]),
        nx=int(problem["_audit_nx"]),
        ny=int(problem["_audit_ny"]),
        poly_order=int(problem["_audit_poly_order"]),
        scalar_order=int(problem["_audit_scalar_order"]),
    )
    mapping = _map_pycutfem_to_fenics_alpha(problem, W_fx)
    w_k_fx, w_n_fx = _load_fenics_state(problem, W_fx, mapping)
    r_fx, a_fx = _fenics_alpha_forms(
        W=W_fx,
        w_k=w_k_fx,
        w_n=w_n_fx,
        dt=float(problem["_audit_dt"]),
        theta=float(problem["_audit_theta"]),
        qdeg=int(qdeg),
        alpha_advect_with=str(alpha_advect_with),
        alpha_advection_form=str(alpha_advection_form),
    )

    form_r_fx = dolfinx.fem.form(r_fx)
    vec_fx = dolfinx.fem.petsc.assemble_vector(form_r_fx)
    r_fx_arr = vec_fx.array.copy()
    form_a_fx = dolfinx.fem.form(a_fx)
    A_fx = dolfinx.fem.petsc.assemble_matrix(form_a_fx)
    A_fx.assemble()
    indptr, indices, data = A_fx.getValuesCSR()
    J_fx = csr_matrix((data, indices, indptr), shape=A_fx.getSize()).tocsr()

    dh = problem["dh"]
    alpha_pc = np.asarray(dh.get_field_slice("alpha"), dtype=int)
    cols_pc = np.concatenate(
        [
            np.asarray(dh.get_field_slice("v_x"), dtype=int),
            np.asarray(dh.get_field_slice("v_y"), dtype=int),
            np.asarray(dh.get_field_slice("vS_x"), dtype=int),
            np.asarray(dh.get_field_slice("vS_y"), dtype=int),
            np.asarray(dh.get_field_slice("alpha"), dtype=int),
        ]
    )
    alpha_fx = np.asarray(mapping["alpha"], dtype=int)
    cols_fx = np.concatenate(
        [
            np.asarray(mapping["v_x"], dtype=int),
            np.asarray(mapping["v_y"], dtype=int),
            np.asarray(mapping["vS_x"], dtype=int),
            np.asarray(mapping["vS_y"], dtype=int),
            np.asarray(mapping["alpha"], dtype=int),
        ]
    )

    _, r_pc = assemble_form(Equation(None, forms.r_alpha), dof_handler=dh, bcs=[], quad_order=int(qdeg), backend=str(backend))
    J_pc, _ = assemble_form(Equation(forms.a_alpha, None), dof_handler=dh, bcs=[], quad_order=int(qdeg), backend=str(backend))
    r_pc = np.asarray(r_pc, dtype=float)[alpha_pc]
    J_pc = J_pc.tocsr()[alpha_pc, :][:, cols_pc].toarray()
    r_fx = np.asarray(r_fx_arr, dtype=float)[alpha_fx]
    J_fx = J_fx[alpha_fx, :][:, cols_fx].toarray()

    res_diff = r_pc - r_fx
    jac_diff = J_pc - J_fx
    res_denom = max(1.0, float(np.linalg.norm(r_pc, ord=np.inf)), float(np.linalg.norm(r_fx, ord=np.inf)))
    jac_denom = max(1.0, float(np.linalg.norm(J_pc, ord=np.inf)), float(np.linalg.norm(J_fx, ord=np.inf)))
    return (
        float(np.max(np.abs(res_diff))),
        float(np.linalg.norm(res_diff, ord=np.inf) / res_denom),
        float(np.max(np.abs(jac_diff))),
        float(np.linalg.norm(jac_diff, ord=np.inf) / jac_denom),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark 7 Jacobian audit against FD and FEniCSx derivative.")
    ap.add_argument("--out", type=Path, default=Path("out/benchmark7_fenics_jacobian_audit"))
    ap.add_argument("--nx", type=int, default=2)
    ap.add_argument("--ny", type=int, default=3)
    ap.add_argument("--poly-order", type=int, default=2)
    ap.add_argument("--pressure-order", type=int, default=1)
    ap.add_argument("--scalar-order", type=int, default=1)
    ap.add_argument("--enable-phi-evolution", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pycutfem-backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--n-random-directions", type=int, default=1)
    ap.add_argument(
        "--cases",
        type=str,
        nargs="+",
        default=("vS_band", "v_band", "interface_band"),
        choices=("vS_band", "v_band", "interface_band"),
    )
    args = ap.parse_args()

    all_cases = [
        ("vS_band", "vS", "interface_band_conservative"),
        ("v_band", "v", "interface_band_conservative"),
        ("interface_band", "interface", "interface_band_conservative"),
    ]
    wanted = set(args.cases)
    cases = [item for item in all_cases if item[0] in wanted]

    summaries: list[CaseSummary] = []
    args.out.mkdir(parents=True, exist_ok=True)
    for name, adv_with, adv_form in cases:
        problem, forms, field_to_func, qdeg = _build_benchmark_case(
            Lx=1.0,
            Ly=1.5,
            nx=int(args.nx),
            ny=int(args.ny),
            poly_order=int(args.poly_order),
            pressure_order=int(args.pressure_order),
            scalar_order=int(args.scalar_order),
            dt=0.01,
            theta=1.0,
            enable_phi_evolution=bool(args.enable_phi_evolution),
            alpha_advect_with=str(adv_with),
            alpha_advection_form=str(adv_form),
        )
        directional_fd_max_rel, directional_fd_per_case = _directional_fd_audit(
            problem,
            forms,
            field_to_func,
            qdeg=int(qdeg),
            enable_phi_evolution=bool(args.enable_phi_evolution),
            backend=str(args.pycutfem_backend),
            n_random=int(args.n_random_directions),
        )
        alpha_res_max_abs, alpha_res_rel, alpha_jac_max_abs, alpha_jac_rel = _pycutfem_alpha_compare(
            problem,
            forms,
            qdeg=int(qdeg),
            alpha_advect_with=str(adv_with),
            alpha_advection_form=str(adv_form),
            backend=str(args.pycutfem_backend),
        )
        summaries.append(
            CaseSummary(
                name=str(name),
                directional_fd_max_rel=float(directional_fd_max_rel),
                directional_fd_per_case={k: float(v) for k, v in directional_fd_per_case.items()},
                alpha_res_max_abs=float(alpha_res_max_abs),
                alpha_res_rel=float(alpha_res_rel),
                alpha_jac_max_abs=float(alpha_jac_max_abs),
                alpha_jac_rel=float(alpha_jac_rel),
            )
        )
        (args.out / "comparison_with_fenics.json").write_text(
            json.dumps([s.__dict__ for s in summaries], indent=2),
            encoding="utf-8",
        )

    data = [s.__dict__ for s in summaries]
    (args.out / "comparison_with_fenics.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    for item in data:
        print(json.dumps(item, indent=2))


if __name__ == "__main__":
    main()
