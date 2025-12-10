import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl
import basix.ufl
from basix.ufl import mixed_element
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix

from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.utils.bitset import BitSet
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    TrialFunction,
    TestFunction,
    VectorTrialFunction,
    VectorTestFunction,
    Function,
    VectorFunction,
    Constant,
    Identity,
    cof,
    det,
    dot,
    grad,
    inner,
    inv,
    trace,
)
from pycutfem.ufl.measures import dx
from pycutfem.ufl.forms import assemble_form, Equation
from pycutfem.fem.mixedelement import MixedElement


# -----------------------------------------------------------------------------#
# Helper utilities                                                             #
# -----------------------------------------------------------------------------#
FIELDS = ["ux", "uy", "dx", "dy", "p"]
I2_PC = Identity(2)
I2_FX = ufl.Identity(2)


def _cof2(A):
    """
    2x2 cofactor without ufl.cofactor (dolfinx UFL may lack it).
    Returns [[a22, -a12], [-a21, a11]].
    """
    return ufl.as_tensor(((A[1, 1], -A[0, 1]), (-A[1, 0], A[0, 0])))


def one_to_one_map_coords(coords_pc: np.ndarray, coords_fx: np.ndarray) -> np.ndarray:
    cost = np.linalg.norm(coords_fx[:, None, :] - coords_pc[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    return rows[np.argsort(cols)]


def get_pc_field_coords(dh: DofHandler, field: str) -> Tuple[np.ndarray, np.ndarray]:
    coords = dh.get_dof_coords(field)
    ids = dh.get_field_slice(field)
    return coords, ids


def get_fx_field_coords(W, field: str):
    # Blocks: 0 -> velocity (vector), 1 -> displacement (vector), 2 -> pressure (scalar)
    block_component = {
        "ux": (0, 0),
        "uy": (0, 1),
        "dx": (1, 0),
        "dy": (1, 1),
        "p": (2, None),
    }[field]
    if block_component[1] is None:
        subspace, parent_map = W.sub(block_component[0]).collapse()
    else:
        subspace, parent_map = W.sub(block_component[0]).sub(block_component[1]).collapse()
    coords = subspace.tabulate_dof_coordinates()[:, :2]
    return coords, np.array(parent_map, dtype=int)


def create_true_dof_map(dh_pc: DofHandler, W_fx) -> np.ndarray:
    P = np.zeros(dh_pc.total_dofs, dtype=int)
    for field in FIELDS:
        pc_coords, pc_ids = get_pc_field_coords(dh_pc, field)
        fx_coords, fx_parent = get_fx_field_coords(W_fx, field)
        perm = one_to_one_map_coords(pc_coords, fx_coords)
        P[pc_ids] = fx_parent[perm]
    return P


def compare_term(name: str, pc_form, fx_form, *, mat: bool, deg: int, P_map: np.ndarray, dh_pc: DofHandler, W_fx, backend: str, rtol=1e-8, atol=1e-8):
    print(f"\n--- Comparing: {name} (deg={deg}, mat={mat}) ---")

    J_pc = R_pc = J_fx = R_fx = None
    compiled_fx = dolfinx.fem.form(fx_form)

    if mat:
        J_pc, _ = assemble_form(Equation(pc_form, None), dof_handler=dh_pc, quad_degree=deg, backend=backend)
        A = dolfinx.fem.petsc.assemble_matrix(compiled_fx)
        A.assemble()
        indptr, indices, data = A.getValuesCSR()
        J_fx = csr_matrix((data, indices, indptr), shape=A.getSize())
    else:
        _, R_pc = assemble_form(Equation(None, pc_form), dof_handler=dh_pc, quad_degree=deg, backend=backend)
        vec = dolfinx.fem.petsc.assemble_vector(compiled_fx)
        R_fx = vec.array

    success = True
    if mat and J_pc is not None and J_fx is not None:
        dense_pc = J_pc.toarray()
        dense_fx = J_fx.toarray()[P_map, :][:, P_map]
        output_dir = "garbage"
        os.makedirs(output_dir, exist_ok=True)
        safe_term_name = name.replace(" ", "_").lower()
        filename = os.path.join(output_dir, f"{safe_term_name}_jacobian.xlsx")
        with pd.ExcelWriter(filename) as writer:
            pd.DataFrame(dense_pc).to_excel(writer, sheet_name="pycutfem", index=False, header=False)
            pd.DataFrame(dense_fx).to_excel(writer, sheet_name="fenics", index=False, header=False)
            pd.DataFrame(np.abs(dense_pc - dense_fx) < 1e-12).to_excel(writer, sheet_name="difference", index=False, header=False)
        print(f"✅ Jacobian matrices saved to '{filename}'")
        try:
            np.testing.assert_allclose(dense_pc, dense_fx, rtol=rtol, atol=atol)
            print("✅ Jacobian matches.")
        except AssertionError as exc:
            success = False
            print("❌ Jacobian mismatch.")
            print(exc)
    if (not mat) and R_pc is not None and R_fx is not None:
        r_pc = R_pc.flatten()
        r_fx = R_fx[P_map]
        output_dir = "garbage"
        os.makedirs(output_dir, exist_ok=True)
        safe_term_name = name.replace(" ", "_").lower()
        filename = os.path.join(output_dir, f"{safe_term_name}_residual.xlsx")
        pd.DataFrame(
            {
                "pc_dof": np.arange(dh_pc.total_dofs),
                "pc_residual": r_pc,
                "fx_residual": r_fx,
                "abs_diff": np.abs(r_pc - r_fx),
            }
        ).to_excel(filename, sheet_name="residual", index=False)
        print(f"✅ Residual comparison data saved to '{filename}'")
        try:
            np.testing.assert_allclose(r_pc, r_fx, rtol=rtol, atol=atol)
            print("✅ Residual matches.")
        except AssertionError as exc:
            success = False
            print("❌ Residual mismatch.")
            print(exc)
    return success


# -----------------------------------------------------------------------------#
# Problem setup                                                                #
# -----------------------------------------------------------------------------#
def setup_pycutfem():
    nx, ny = 2, 1
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=2)
    mesh_pc = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=2)
    mesh_pc.tag_boundary_edges({"outlet": lambda x, y: np.isclose(x, 1.0)})

    centers = mesh_pc.nodes_x_y_pos[mesh_pc.corner_connectivity].mean(axis=1)
    fluid_mask = centers[:, 0] < 0.5
    solid_mask = ~fluid_mask
    fluid_bs = BitSet(fluid_mask)
    solid_bs = BitSet(solid_mask)
    outlet_bs = mesh_pc.edge_bitset("outlet")

    element = MixedElement(mesh_pc, field_specs={"ux": 2, "uy": 2, "dx": 2, "dy": 2, "p": 1})
    dh = DofHandler(element, method="cg")
    vel_space = FunctionSpace(name="vel", field_names=["ux", "uy"], dim=1)
    disp_space = FunctionSpace(name="disp", field_names=["dx", "dy"], dim=1)

    du = VectorTrialFunction(space=vel_space, dof_handler=dh)
    dd = VectorTrialFunction(space=disp_space, dof_handler=dh)
    dp = TrialFunction(name="p_trial", field_name="p", dof_handler=dh)

    v = VectorTestFunction(space=vel_space, dof_handler=dh)
    w = VectorTestFunction(space=disp_space, dof_handler=dh)
    q = TestFunction(name="p_test", field_name="p", dof_handler=dh)

    uk = VectorFunction(name="u", field_names=["ux", "uy"], dof_handler=dh)
    u_prev = VectorFunction(name="u_prev", field_names=["ux", "uy"], dof_handler=dh)
    dk = VectorFunction(name="d", field_names=["dx", "dy"], dof_handler=dh)
    d_prev = VectorFunction(name="d_prev", field_names=["dx", "dy"], dof_handler=dh)
    pk = Function(name="p", field_name="p", dof_handler=dh)
    p_prev = Function(name="p_prev", field_name="p", dof_handler=dh)

    nu_s = 0.4
    rho_f = Constant(1.0e3)
    nu_f = Constant(1.0e-3)
    mu_f = rho_f * nu_f
    rho_s = Constant(1.0e3)
    mu_s = Constant(0.5e6)
    E_s_val = 2.0 * float(mu_s.value) * (1.0 + nu_s)
    lambda_s = Constant(E_s_val * nu_s / ((1.0 + nu_s) * (1.0 - 2.0 * nu_s)))
    pc = {
        "mesh": mesh_pc,
        "fluid_bs": fluid_bs,
        "solid_bs": solid_bs,
        "outlet_bs": outlet_bs,
        "du": du,
        "dd": dd,
        "dp": dp,
        "v": v,
        "w": w,
        "q": q,
        "uk": uk,
        "u_prev": u_prev,
        "dk": dk,
        "d_prev": d_prev,
        "pk": pk,
        "p_prev": p_prev,
        "rho_f": rho_f,
        "mu_f": mu_f,
        "rho_s": rho_s,
        "lambda_s": lambda_s,
        "mu_s": mu_s,
        "alpha_u": Constant(1.0e-6),
        "stab_eps": Constant(1.0e-8),
        "dt": Constant(0.1),
        "theta": Constant(0.6),
        "quad_order": 8,
        "dof_handler": dh,
    }
    return pc


def setup_fenics():
    mesh_fx = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [(0.0, 0.0), (1.0, 1.0)], [2, 1], cell_type=dolfinx.mesh.CellType.quadrilateral)
    num_cells = mesh_fx.topology.index_map(mesh_fx.topology.dim).size_local
    mesh_fx.topology.create_connectivity(mesh_fx.topology.dim, 0)
    cell_vertices = mesh_fx.topology.connectivity(mesh_fx.topology.dim, 0).array.reshape(num_cells, -1)
    coords = mesh_fx.geometry.x
    centers = coords[cell_vertices].mean(axis=1)
    markers = np.where(centers[:, 0] < 0.5, 0, 1).astype(np.int32)
    cell_tags = dolfinx.mesh.meshtags(mesh_fx, mesh_fx.topology.dim, np.arange(num_cells, dtype=np.int32), markers)
    dx_sub = ufl.Measure("dx", domain=mesh_fx, subdomain_data=cell_tags, metadata={"quadrature_degree": 8})

    V2 = basix.ufl.element("Lagrange", "quadrilateral", 2, shape=(mesh_fx.geometry.dim,))
    Vp = basix.ufl.element("Lagrange", "quadrilateral", 1)
    W_el = mixed_element([V2, V2, Vp])
    W = dolfinx.fem.functionspace(mesh_fx, W_el)

    trial = ufl.TrialFunction(W)
    test = ufl.TestFunction(W)
    du_fx, dd_fx, dp_fx = ufl.split(trial)
    v_fx, w_fx, q_fx = ufl.split(test)

    state_k = dolfinx.fem.Function(W, name="state_k")
    state_prev = dolfinx.fem.Function(W, name="state_prev")
    u_k_fx, d_k_fx, p_k_fx = ufl.split(state_k)
    u_prev_fx, d_prev_fx, p_prev_fx = ufl.split(state_prev)

    nu_s = 0.4
    rho_f = dolfinx.fem.Constant(mesh_fx, 1.0e3)
    nu_f = dolfinx.fem.Constant(mesh_fx, 1.0e-3)
    mu_f = dolfinx.fem.Constant(mesh_fx, float(rho_f.value) * float(nu_f.value))
    rho_s = dolfinx.fem.Constant(mesh_fx, 1.0e3)
    mu_s = dolfinx.fem.Constant(mesh_fx, 0.5e6)
    E_s = 2.0 * float(mu_s.value) * (1.0 + nu_s)
    lambda_s = dolfinx.fem.Constant(mesh_fx, E_s * nu_s / ((1.0 + nu_s) * (1.0 - 2.0 * nu_s)))

    fx = {
        "mesh": mesh_fx,
        "dx": dx_sub,
        "cell_tags": cell_tags,
        "W": W,
        "trial": (du_fx, dd_fx, dp_fx),
        "test": (v_fx, w_fx, q_fx),
        "state_k": state_k,
        "state_prev": state_prev,
        "u_k": u_k_fx,
        "d_k": d_k_fx,
        "p_k": p_k_fx,
        "u_prev": u_prev_fx,
        "d_prev": d_prev_fx,
        "p_prev": p_prev_fx,
        "rho_f": rho_f,
        "mu_f": mu_f,
        "rho_s": rho_s,
        "lambda_s": lambda_s,
        "mu_s": mu_s,
        "alpha_u": dolfinx.fem.Constant(mesh_fx, 1.0e-6),
        "stab_eps": dolfinx.fem.Constant(mesh_fx, 1.0e-8),
        "dt": dolfinx.fem.Constant(mesh_fx, 0.1),
        "theta": dolfinx.fem.Constant(mesh_fx, 0.6),
    }
    return fx


# -----------------------------------------------------------------------------#
# Initialization                                                               #
# -----------------------------------------------------------------------------#
def _vel_init(x, y):
    return np.vstack((1.0 + 0.25 * x + 0.1 * y, -0.3 + 0.15 * x * y))


def _disp_init(x, y):
    return np.vstack((0.05 * x, -0.02 * y))


def _pres_init(x, y):
    return 0.25 + 0.1 * x - 0.05 * y


def initialize_pycutfem(pc: Dict):
    pc["uk"].set_values_from_function(lambda x, y: _vel_init(x, y))
    pc["u_prev"].set_values_from_function(lambda x, y: 0.7 * _vel_init(x, y))
    pc["dk"].set_values_from_function(lambda x, y: _disp_init(x, y))
    pc["d_prev"].set_values_from_function(lambda x, y: 0.5 * _disp_init(x, y))
    pc["pk"].set_values_from_function(lambda x, y: _pres_init(x, y))
    pc["p_prev"].set_values_from_function(lambda x, y: 0.8 * _pres_init(x, y))


def _fx_interpolator(func):
    def wrapper(x):
        vals = func(x[0], x[1])
        arr = np.atleast_2d(vals)
        return arr
    return wrapper


def initialize_fenics(fx: Dict):
    W = fx["W"]
    state_k = fx["state_k"]
    state_prev = fx["state_prev"]

    sub_u, map_u = W.sub(0).collapse()
    sub_d, map_d = W.sub(1).collapse()
    sub_p, map_p = W.sub(2).collapse()

    u_fun = dolfinx.fem.Function(sub_u)
    u_fun.interpolate(_fx_interpolator(_vel_init))
    d_fun = dolfinx.fem.Function(sub_d)
    d_fun.interpolate(_fx_interpolator(_disp_init))
    p_fun = dolfinx.fem.Function(sub_p)
    p_fun.interpolate(_fx_interpolator(_pres_init))

    state_k.x.array[map_u] = u_fun.x.array
    state_k.x.array[map_d] = d_fun.x.array
    state_k.x.array[map_p] = p_fun.x.array
    state_k.x.scatter_forward()

    u_prev_fun = dolfinx.fem.Function(sub_u)
    u_prev_fun.interpolate(_fx_interpolator(lambda x, y: 0.7 * _vel_init(x, y)))
    d_prev_fun = dolfinx.fem.Function(sub_d)
    d_prev_fun.interpolate(_fx_interpolator(lambda x, y: 0.5 * _disp_init(x, y)))
    p_prev_fun = dolfinx.fem.Function(sub_p)
    p_prev_fun.interpolate(_fx_interpolator(lambda x, y: 0.8 * _pres_init(x, y)))

    state_prev.x.array[map_u] = u_prev_fun.x.array
    state_prev.x.array[map_d] = d_prev_fun.x.array
    state_prev.x.array[map_p] = p_prev_fun.x.array
    state_prev.x.scatter_forward()


# -----------------------------------------------------------------------------#
# Form construction                                                            #
# -----------------------------------------------------------------------------#
def _structure_stress_pc(F, mu_s, lambda_s):
    E = 0.5 * (dot(F.T, F) - I2_PC)
    S = lambda_s * trace(E) * I2_PC + Constant(2.0) * mu_s * E
    J = det(F)
    return (1.0 / J) * dot(dot(F, S), F.T)


def _structure_stress_fx(F, mu_s, lambda_s):
    E = 0.5 * (ufl.dot(F.T, F) - I2_FX)
    S = lambda_s * ufl.tr(E) * I2_FX + 2.0 * mu_s * E
    J = ufl.det(F)
    return (1.0 / J) * ufl.dot(ufl.dot(F, S), F.T)


def _structure_tangent_pc(F, grad_dd, mu_s, lambda_s):
    delta_F = grad_dd
    E = 0.5 * (dot(F.T, F) - I2_PC)
    S = lambda_s * trace(E) * I2_PC + Constant(2.0) * mu_s * E
    delta_E = 0.5 * (dot(delta_F.T, F) + dot(F.T, delta_F))
    delta_S = lambda_s * trace(delta_E) * I2_PC + Constant(2.0) * mu_s * delta_E
    return dot(delta_F, S) + dot(F, delta_S)


def _structure_tangent_fx(F, grad_dd, mu_s, lambda_s):
    delta_F = grad_dd
    E = 0.5 * (ufl.dot(F.T, F) - I2_FX)
    S = lambda_s * ufl.tr(E) * I2_FX + 2.0 * mu_s * E
    delta_E = 0.5 * (ufl.dot(delta_F.T, F) + ufl.dot(F.T, delta_F))
    delta_S = lambda_s * ufl.tr(delta_E) * I2_FX + 2.0 * mu_s * delta_E
    return ufl.dot(delta_F, S) + ufl.dot(F, delta_S)


def build_terms(pc: Dict, fx: Dict):
    def dx_pc_f(deg):
        return dx(defined_on=pc["fluid_bs"], metadata={"q": deg})

    def dx_pc_s(deg):
        return dx(defined_on=pc["solid_bs"], metadata={"q": deg})

    def dx_fx_sub(deg, subdomain: int):
        measure = ufl.Measure(
            "dx",
            domain=fx["mesh"],
            subdomain_data=fx["cell_tags"],
            metadata={"quadrature_degree": deg},
        )
        return measure(subdomain)

    du_pc, dd_pc, dp_pc = pc["du"], pc["dd"], pc["dp"]
    v_pc, w_pc, q_pc = pc["v"], pc["w"], pc["q"]
    uk_pc, u_prev_pc = pc["uk"], pc["u_prev"]
    dk_pc, d_prev_pc = pc["dk"], pc["d_prev"]
    pk_pc, p_prev_pc = pc["pk"], pc["p_prev"]

    du_fx, dd_fx, dp_fx = fx["trial"]
    v_fx, w_fx, q_fx = fx["test"]
    uk_fx, u_prev_fx = fx["u_k"], fx["u_prev"]
    dk_fx, d_prev_fx = fx["d_k"], fx["d_prev"]
    pk_fx, p_prev_fx = fx["p_k"], fx["p_prev"]

    rho_f_pc, mu_f_pc = pc["rho_f"], pc["mu_f"]
    rho_s_pc, mu_s_pc, lambda_s_pc = pc["rho_s"], pc["mu_s"], pc["lambda_s"]
    dt_pc, theta_pc = pc["dt"], pc["theta"]
    alpha_u_pc = pc["alpha_u"]

    rho_f_fx, mu_f_fx = fx["rho_f"], fx["mu_f"]
    rho_s_fx, mu_s_fx, lambda_s_fx = fx["rho_s"], fx["mu_s"], fx["lambda_s"]
    dt_fx, theta_fx = fx["dt"], fx["theta"]
    alpha_u_fx = fx["alpha_u"]

    # Geometry (current + previous)
    F_pc = I2_PC + grad(dk_pc)
    F_prev_pc = I2_PC + grad(d_prev_pc)
    J_pc = det(F_pc)
    J_prev_pc = det(F_prev_pc)
    Finv_pc = inv(F_pc)
    Finv_prev_pc = inv(F_prev_pc)
    grad_uk_pc = grad(uk_pc)
    grad_u_prev_pc = grad(u_prev_pc)
    pI_pc = pk_pc * I2_PC

    F_fx = I2_FX + ufl.grad(dk_fx)
    F_prev_fx = I2_FX + ufl.grad(d_prev_fx)
    J_fx = ufl.det(F_fx)
    J_prev_fx = ufl.det(F_prev_fx)
    Finv_fx = ufl.inv(F_fx)
    Finv_prev_fx = ufl.inv(F_prev_fx)
    grad_uk_fx = ufl.grad(uk_fx)
    grad_u_prev_fx = ufl.grad(u_prev_fx)
    pI_fx = pk_fx * I2_FX

    # Linearization helpers
    grad_dd_pc = grad(dd_pc)
    J_LinU_pc = inner(cof(F_pc), grad_dd_pc)
    Finv_LinU_pc = -dot(Finv_pc, dot(grad_dd_pc, Finv_pc))
    J_F_inv_T_LinU_pc = cof(grad_dd_pc)

    grad_dd_fx = ufl.grad(dd_fx)
    J_LinU_fx = ufl.inner(_cof2(F_fx), grad_dd_fx)
    Finv_LinU_fx = -ufl.dot(Finv_fx, ufl.dot(grad_dd_fx, Finv_fx))
    J_F_inv_T_LinU_fx = _cof2(grad_dd_fx)

    # Convenience
    grad_du_pc = grad(du_pc)
    grad_du_fx = ufl.grad(du_fx)
    grad_dk_pc = grad(dk_pc)

    # ---------------- Residual terms (fluid) ----------------
    def acc_res_pc(deg):
        return rho_f_pc * Constant(0.5) * (J_pc + J_prev_pc) * inner(uk_pc - u_prev_pc, v_pc) * dx_pc_f(deg)

    def acc_res_fx(deg):
        return rho_f_fx * 0.5 * (J_fx + J_prev_fx) * ufl.dot(uk_fx - u_prev_fx, v_fx) * dx_fx_sub(deg, 0)

    def conv_new_res_pc(deg):
        return dt_pc * theta_pc * dot(rho_f_pc * J_pc * dot(dot(grad_uk_pc, Finv_pc), uk_pc), v_pc) * dx_pc_f(deg)

    def conv_new_res_fx(deg):
        return dt_fx * theta_fx * ufl.dot(rho_f_fx * J_fx * ufl.dot(ufl.dot(grad_uk_fx, Finv_fx), uk_fx), v_fx) * dx_fx_sub(deg, 0)

    def conv_old_res_pc(deg):
        return dt_pc * (Constant(1.0) - theta_pc) * dot(rho_f_pc * J_prev_pc * dot(dot(grad_u_prev_pc, Finv_prev_pc), u_prev_pc), v_pc) * dx_pc_f(deg)

    def conv_old_res_fx(deg):
        return dt_fx * (1.0 - theta_fx) * ufl.dot(rho_f_fx * J_prev_fx * ufl.dot(ufl.dot(grad_u_prev_fx, Finv_prev_fx), u_prev_fx), v_fx) * dx_fx_sub(deg, 0)

    def conv_mesh_res_pc(deg):
        return -dot(rho_f_pc * J_pc * dot(dot(grad_uk_pc, Finv_pc), dk_pc - d_prev_pc), v_pc) * dx_pc_f(deg)

    def conv_mesh_res_fx(deg):
        return -ufl.dot(rho_f_fx * J_fx * ufl.dot(ufl.dot(grad_uk_fx, Finv_fx), dk_fx - d_prev_fx), v_fx) * dx_fx_sub(deg, 0)

    fluid_pressure_pc = -(J_pc * dot(pI_pc, Finv_pc.T))
    fluid_pressure_fx = -(J_fx * ufl.dot(pI_fx, Finv_fx.T))

    def pressure_res_pc(deg):
        return dt_pc * inner(fluid_pressure_pc, grad(v_pc)) * dx_pc_f(deg)

    def pressure_res_fx(deg):
        return dt_fx * ufl.inner(fluid_pressure_fx, ufl.grad(v_fx)) * dx_fx_sub(deg, 0)

    sigma_ALE_pc = mu_f_pc * (dot(grad_uk_pc, Finv_pc) + dot(Finv_pc.T, grad_uk_pc.T))
    sigma_ALE_prev_pc = mu_f_pc * (dot(grad_u_prev_pc, Finv_prev_pc) + dot(Finv_prev_pc.T, grad_u_prev_pc.T))
    def visc_new_pc(deg):
        return dt_pc * theta_pc * inner(J_pc * dot(sigma_ALE_pc, Finv_pc.T), grad(v_pc)) * dx_pc_f(deg)

    def visc_old_pc(deg):
        return dt_pc * (Constant(1.0) - theta_pc) * inner(J_prev_pc * dot(sigma_ALE_prev_pc, Finv_prev_pc.T), grad(v_pc)) * dx_pc_f(deg)

    sigma_ALE_fx = mu_f_fx * (ufl.dot(grad_uk_fx, Finv_fx) + ufl.dot(Finv_fx.T, grad_uk_fx.T))
    sigma_ALE_prev_fx = mu_f_fx * (ufl.dot(grad_u_prev_fx, Finv_prev_fx) + ufl.dot(Finv_prev_fx.T, grad_u_prev_fx.T))
    def visc_new_fx(deg):
        return dt_fx * theta_fx * ufl.inner(J_fx * ufl.dot(sigma_ALE_fx, Finv_fx.T), ufl.grad(v_fx)) * dx_fx_sub(deg, 0)

    def visc_old_fx(deg):
        return dt_fx * (1.0 - theta_fx) * ufl.inner(J_prev_fx * ufl.dot(sigma_ALE_prev_fx, Finv_prev_fx.T), ufl.grad(v_fx)) * dx_fx_sub(deg, 0)

    def biharmonic_res_pc(deg):
        return (alpha_u_pc / J_pc) * inner(grad_dk_pc, grad(w_pc)) * dx_pc_f(deg)

    def biharmonic_res_fx(deg):
        return (alpha_u_fx / J_fx) * ufl.inner(ufl.grad(dk_fx), ufl.grad(w_fx)) * dx_fx_sub(deg, 0)

    def incompressibility_res_pc(deg):
        return inner(cof(F_pc), grad(uk_pc)) * q_pc * dx_pc_f(deg)

    def incompressibility_res_fx(deg):
        return ufl.inner(_cof2(F_fx), ufl.grad(uk_fx)) * q_fx * dx_fx_sub(deg, 0)

    # ---------------- Residual terms (solid) ----------------
    solid_stress_pc = _structure_stress_pc(F_pc, mu_s_pc, lambda_s_pc)
    solid_stress_prev_pc = _structure_stress_pc(F_prev_pc, mu_s_pc, lambda_s_pc)
    def solid_res_pc(deg):
        return (
            rho_s_pc * inner(uk_pc - u_prev_pc, v_pc)
            + dt_pc * theta_pc * inner(J_pc * dot(solid_stress_pc, Finv_pc.T), grad(v_pc))
            + dt_pc * (Constant(1.0) - theta_pc) * inner(J_prev_pc * dot(solid_stress_prev_pc, Finv_prev_pc.T), grad(v_pc))
            + rho_s_pc * inner(dk_pc - d_prev_pc, w_pc)
            - rho_s_pc * dt_pc * theta_pc * inner(uk_pc, w_pc)
            - rho_s_pc * dt_pc * (Constant(1.0) - theta_pc) * inner(u_prev_pc, w_pc)
            + pk_pc * q_pc
        ) * dx_pc_s(deg)

    solid_stress_fx = _structure_stress_fx(F_fx, mu_s_fx, lambda_s_fx)
    solid_stress_prev_fx = _structure_stress_fx(F_prev_fx, mu_s_fx, lambda_s_fx)
    def solid_res_fx(deg):
        return (
            rho_s_fx * ufl.dot(uk_fx - u_prev_fx, v_fx)
            + dt_fx * theta_fx * ufl.inner(J_fx * ufl.dot(solid_stress_fx, Finv_fx.T), ufl.grad(v_fx))
            + dt_fx * (1.0 - theta_fx) * ufl.inner(J_prev_fx * ufl.dot(solid_stress_prev_fx, Finv_prev_fx.T), ufl.grad(v_fx))
            + rho_s_fx * ufl.dot(dk_fx - d_prev_fx, w_fx)
            - rho_s_fx * dt_fx * theta_fx * ufl.dot(uk_fx, w_fx)
            - rho_s_fx * dt_fx * (1.0 - theta_fx) * ufl.dot(u_prev_fx, w_fx)
            + pk_fx * q_fx
        ) * dx_fx_sub(deg, 1)

    # ---------------- Jacobian terms (fluid) ----------------
    def acc_jac_pc(deg):
        return Constant(0.5) * rho_f_pc * dot(J_LinU_pc * (uk_pc - u_prev_pc) + (J_pc + J_prev_pc) * du_pc, v_pc) * dx_pc_f(deg)

    def acc_jac_fx(deg):
        return 0.5 * rho_f_fx * ufl.dot(J_LinU_fx * (uk_fx - u_prev_fx) + (J_fx + J_prev_fx) * du_fx, v_fx) * dx_fx_sub(deg, 0)

    grad_uk_Finv_pc = dot(grad_uk_pc, Finv_pc)
    grad_uk_Finv_fx = ufl.dot(grad_uk_fx, Finv_fx)
    stress_ALE_pc = -pI_pc + mu_f_pc * (grad_uk_Finv_pc + dot(Finv_pc.T, grad_uk_pc.T))
    stress_ALE_fx = -pI_fx + mu_f_fx * (grad_uk_Finv_fx + ufl.dot(Finv_fx.T, grad_uk_fx.T))

    def conv_v_jac_pc(deg):
        expr = rho_f_pc * (
            J_LinU_pc * dot(grad_uk_Finv_pc, uk_pc)
            + J_pc * dot(dot(grad_uk_pc, Finv_LinU_pc), uk_pc)
            + J_pc * dot(dot(grad_du_pc, Finv_pc), uk_pc)
            + J_pc * dot(grad_uk_Finv_pc, du_pc)
        )
        return dt_pc * theta_pc * dot(expr, v_pc) * dx_pc_f(deg)

    def conv_v_jac_fx(deg):
        expr = rho_f_fx * (
            J_LinU_fx * ufl.dot(grad_uk_Finv_fx, uk_fx)
            + J_fx * ufl.dot(ufl.dot(grad_uk_fx, Finv_LinU_fx), uk_fx)
            + J_fx * ufl.dot(ufl.dot(grad_du_fx, Finv_fx), uk_fx)
            + J_fx * ufl.dot(grad_uk_Finv_fx, du_fx)
        )
        return dt_fx * theta_fx * ufl.dot(expr, v_fx) * dx_fx_sub(deg, 0)

    def conv_d_jac_pc(deg):
        expr = rho_f_pc * (
            J_LinU_pc * dot(grad_uk_Finv_pc, dk_pc)
            + J_pc * dot(dot(grad_uk_pc, Finv_LinU_pc), dk_pc)
            + J_pc * dot(grad_uk_Finv_pc, dd_pc)
            + J_pc * dot(dot(grad_du_pc, Finv_pc), dk_pc)
        )
        return -dot(expr, v_pc) * dx_pc_f(deg)

    def conv_d_jac_fx(deg):
        expr = rho_f_fx * (
            J_LinU_fx * ufl.dot(grad_uk_Finv_fx, dk_fx)
            + J_fx * ufl.dot(ufl.dot(grad_uk_fx, Finv_LinU_fx), dk_fx)
            + J_fx * ufl.dot(grad_uk_Finv_fx, dd_fx)
            + J_fx * ufl.dot(ufl.dot(grad_du_fx, Finv_fx), dk_fx)
        )
        return -ufl.dot(expr, v_fx) * dx_fx_sub(deg, 0)

    def conv_old_jac_pc(deg):
        expr = rho_f_pc * (
            J_LinU_pc * dot(grad_uk_Finv_pc, d_prev_pc)
            + J_pc * dot(dot(grad_uk_pc, Finv_LinU_pc), d_prev_pc)
            + J_pc * dot(dot(grad_du_pc, Finv_pc), d_prev_pc)
        )
        return dot(expr, v_pc) * dx_pc_f(deg)

    def conv_old_jac_fx(deg):
        expr = rho_f_fx * (
            J_LinU_fx * ufl.dot(grad_uk_Finv_fx, d_prev_fx)
            + J_fx * ufl.dot(ufl.dot(grad_uk_fx, Finv_LinU_fx), d_prev_fx)
            + J_fx * ufl.dot(ufl.dot(grad_du_fx, Finv_fx), d_prev_fx)
        )
        return ufl.dot(expr, v_fx) * dx_fx_sub(deg, 0)

    def stress_term1_common(dp_sym, pI_sym, J_sym, Finv_sym, J_F_inv_T_LinU_sym, *, as_ufl=False):
        return -J_sym * (ufl.dot(dp_sym * I2_FX, Finv_sym.T) if as_ufl else dot(dp_sym * I2_PC, Finv_sym.T)) - (
            ufl.dot(pI_sym, J_F_inv_T_LinU_sym) if as_ufl else dot(pI_sym, J_F_inv_T_LinU_sym)
        )

    stress_term1_pc = stress_term1_common(dp_pc, pI_pc, J_pc, Finv_pc, J_F_inv_T_LinU_pc, as_ufl=False)
    stress_term1_fx = stress_term1_common(dp_fx, pI_fx, J_fx, Finv_fx, J_F_inv_T_LinU_fx, as_ufl=True)

    sigma_LinV_pc = dot(grad_du_pc, Finv_pc) + dot(Finv_pc.T, grad_du_pc.T)
    sigma_LinU_pc = dot(grad_uk_pc, Finv_LinU_pc) + dot(Finv_LinU_pc.T, grad_uk_pc.T)
    stress_term2_pc = mu_f_pc * (J_pc * dot(sigma_LinV_pc + sigma_LinU_pc, Finv_pc.T) + dot(stress_ALE_pc, J_F_inv_T_LinU_pc))

    sigma_LinV_fx = ufl.dot(grad_du_fx, Finv_fx) + ufl.dot(Finv_fx.T, grad_du_fx.T)
    sigma_LinU_fx = ufl.dot(grad_uk_fx, Finv_LinU_fx) + ufl.dot(Finv_LinU_fx.T, grad_uk_fx.T)
    stress_term2_fx = mu_f_fx * (J_fx * ufl.dot(sigma_LinV_fx + sigma_LinU_fx, Finv_fx.T) + ufl.dot(stress_ALE_fx, J_F_inv_T_LinU_fx))

    def diff_term_p_pc(deg):
        return dt_pc * inner(stress_term1_pc, grad(v_pc)) * dx_pc_f(deg)

    def diff_term_p_fx(deg):
        return dt_fx * ufl.inner(stress_term1_fx, ufl.grad(v_fx)) * dx_fx_sub(deg, 0)

    def diff_term_visc_pc(deg):
        return dt_pc * theta_pc * inner(stress_term2_pc, grad(v_pc)) * dx_pc_f(deg)

    def diff_term_visc_fx(deg):
        return dt_fx * theta_fx * ufl.inner(stress_term2_fx, ufl.grad(v_fx)) * dx_fx_sub(deg, 0)

    def bih_geo_pc(deg):
        return (-alpha_u_pc / (J_pc * J_pc) * J_LinU_pc * inner(grad_dk_pc, grad(w_pc))) * dx_pc_f(deg)

    def bih_trial_pc(deg):
        return (alpha_u_pc / J_pc) * inner(grad_dd_pc, grad(w_pc)) * dx_pc_f(deg)

    def bih_geo_fx(deg):
        return (-alpha_u_fx / (J_fx * J_fx) * J_LinU_fx * ufl.inner(ufl.grad(dk_fx), ufl.grad(w_fx))) * dx_fx_sub(deg, 0)

    def bih_trial_fx(deg):
        return (alpha_u_fx / J_fx) * ufl.inner(ufl.grad(dd_fx), ufl.grad(w_fx)) * dx_fx_sub(deg, 0)

    def incompress_jac_pc(deg):
        return (inner(cof(F_pc), grad(du_pc)) + inner(cof(grad_dd_pc), grad_uk_pc)) * q_pc * dx_pc_f(deg)

    def incompress_jac_fx(deg):
        return (ufl.inner(_cof2(F_fx), ufl.grad(du_fx)) + ufl.inner(_cof2(grad_dd_fx), grad_uk_fx)) * q_fx * dx_fx_sub(deg, 0)

    solid_tangent_pc = _structure_tangent_pc(F_pc, grad_dd_pc, mu_s_pc, lambda_s_pc)
    solid_tangent_fx = _structure_tangent_fx(F_fx, grad_dd_fx, mu_s_fx, lambda_s_fx)

    def solid_jac_pc(deg):
        return (
            rho_s_pc * dot(du_pc, v_pc)
            + dt_pc * theta_pc * inner(solid_tangent_pc, grad(v_pc))
            + rho_s_pc * dot(dd_pc, w_pc)
            - dt_pc * theta_pc * dot(du_pc, w_pc)
            + dp_pc * q_pc
        ) * dx_pc_s(deg)

    def solid_jac_fx(deg):
        return (
            rho_s_fx * ufl.dot(du_fx, v_fx)
            + dt_fx * theta_fx * ufl.inner(solid_tangent_fx, ufl.grad(v_fx))
            + rho_s_fx * ufl.dot(dd_fx, w_fx)
            - dt_fx * theta_fx * ufl.dot(du_fx, w_fx)
            + dp_fx * q_fx
        ) * dx_fx_sub(deg, 1)

    qd = pc["quad_order"]

    def term(pc_form, fx_lambda, mat: bool, deg=None):
        return {"pc": pc_form, "f_lambda": fx_lambda, "mat": mat, "deg": deg if deg is not None else qd}

    terms = {
        "res: fluid mass": term(acc_res_pc(qd), acc_res_fx, False, qd),
        "res: fluid convection (new)": term(conv_new_res_pc(qd), conv_new_res_fx, False, qd),
        "res: fluid convection (old)": term(conv_old_res_pc(qd), conv_old_res_fx, False, qd),
        "res: fluid convection (mesh)": term(conv_mesh_res_pc(qd), conv_mesh_res_fx, False, qd),
        "res: fluid pressure": term(pressure_res_pc(qd), pressure_res_fx, False, qd),
        "res: fluid viscous (new)": term(visc_new_pc(qd), visc_new_fx, False, qd),
        "res: fluid viscous (old)": term(visc_old_pc(qd), visc_old_fx, False, qd),
        "res: biharmonic": term(biharmonic_res_pc(qd), biharmonic_res_fx, False, qd),
        "res: incompressibility": term(incompressibility_res_pc(qd), incompressibility_res_fx, False, qd),
        "res: solid block": term(solid_res_pc(qd), solid_res_fx, False, qd),
        "res: full": term(
            acc_res_pc(qd)
            + conv_new_res_pc(qd)
            + conv_old_res_pc(qd)
            + conv_mesh_res_pc(qd)
            + pressure_res_pc(qd)
            + visc_new_pc(qd)
            + visc_old_pc(qd)
            + biharmonic_res_pc(qd)
            + incompressibility_res_pc(qd)
            + solid_res_pc(qd),
            lambda deg: acc_res_fx(deg)
            + conv_new_res_fx(deg)
            + conv_old_res_fx(deg)
            + conv_mesh_res_fx(deg)
            + pressure_res_fx(deg)
            + visc_new_fx(deg)
            + visc_old_fx(deg)
            + biharmonic_res_fx(deg)
            + incompressibility_res_fx(deg)
            + solid_res_fx(deg),
            False,
            qd,
        ),
        "jac: mass": term(acc_jac_pc(qd), acc_jac_fx, True, qd),
        "jac: convection du": term(conv_v_jac_pc(qd), conv_v_jac_fx, True, qd),
        "jac: convection dd": term(conv_d_jac_pc(qd), conv_d_jac_fx, True, qd),
        "jac: convection old disp": term(conv_old_jac_pc(qd), conv_old_jac_fx, True, qd),
        "jac: diffusion p": term(diff_term_p_pc(qd-2), diff_term_p_fx, True, qd+5),
        "jac: diffusion visc": term(diff_term_visc_pc(qd), diff_term_visc_fx, True, qd),
        "jac: biharmonic geom": term(bih_geo_pc(qd), bih_geo_fx, True, qd),
        "jac: biharmonic trial": term(bih_trial_pc(qd), bih_trial_fx, True, qd),
        "jac: incompressibility": term(incompress_jac_pc(qd), incompress_jac_fx, True, qd),
        "jac: solid": term(solid_jac_pc(qd), solid_jac_fx, True, qd),
        "jac: full": term(
            acc_jac_pc(qd)
            + conv_v_jac_pc(qd)
            + conv_d_jac_pc(qd)
            + conv_old_jac_pc(qd)
            + diff_term_p_pc(qd)
            + diff_term_visc_pc(qd)
            + bih_geo_pc(qd)
            + bih_trial_pc(qd)
            + incompress_jac_pc(qd)
            + solid_jac_pc(qd),
            lambda deg: acc_jac_fx(deg)
            + conv_v_jac_fx(deg)
            + conv_d_jac_fx(deg)
            + conv_old_jac_fx(deg)
            + diff_term_p_fx(deg)
            + diff_term_visc_fx(deg)
            + bih_geo_fx(deg)
            + bih_trial_fx(deg)
            + incompress_jac_fx(deg)
            + solid_jac_fx(deg),
            True,
            qd,
        ),
    }
    return terms


# -----------------------------------------------------------------------------#
# Driver                                                                       #
# -----------------------------------------------------------------------------#
def main():
    backend = os.environ.get("BACKEND", "jit")
    pc = setup_pycutfem()
    dh_pc = pc["dof_handler"]
    fx = setup_fenics()

    P_map = create_true_dof_map(dh_pc, fx["W"])
    initialize_pycutfem(pc)
    initialize_fenics(fx)

    terms = build_terms(pc, fx)
    filter_terms = os.environ.get("COMP_FENICS_TERMS")
    if filter_terms:
        allowed = {t.strip() for t in filter_terms.split(",") if t.strip()}
        terms = {k: v for k, v in terms.items() if k in allowed}
        print(f"Filtering to: {sorted(terms)}")

    failed = []
    for name, forms in terms.items():
        deg = forms.get("deg", pc["quad_order"])
        pc_form = forms["pc"]
        fx_form = forms["f_lambda"](deg)
        ok = compare_term(name, pc_form, fx_form, mat=forms["mat"], deg=deg, P_map=P_map, dh_pc=dh_pc, W_fx=fx["W"], backend=backend)
        if not ok:
            failed.append(name)

    print("\n" + "=" * 60)
    print(f"Total terms: {len(terms)}, passed: {len(terms) - len(failed)}, failed: {len(failed)}")
    if failed:
        print("Failed terms:")
        for t in failed:
            print(f"  - {t}")
    print("=" * 60)


if __name__ == "__main__":
    main()
