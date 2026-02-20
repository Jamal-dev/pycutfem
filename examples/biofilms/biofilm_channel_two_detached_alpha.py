"""
Channel flow with two *detached* biofilm blobs represented by a **single** alpha field.

This example is meant as a diagnostic for "post-sloughing" states where the biofilm
indicator α has multiple disconnected components. It uses the one-domain diffuse-
interface model (Navier–Stokes–Brinkman–Biot) and transports α via an Eulerian
reference map:

    α(x,t) = α0(x - u(x,t)),

where u is the skeleton displacement / reference-map field. The intent is to test
Newton convergence in the presence of multiple detached components using the C++
backend.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver, TimeStepperParameters
from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import ds, dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad

from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms


def _tag_rectangle_boundaries(mesh: Mesh, *, L: float, H: float, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - float(L)) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - float(H)) <= tol,
        }
    )


def _smooth_step(z: np.ndarray) -> np.ndarray:
    # Robust sigmoid: 0.5*(1+tanh(z)).
    return 0.5 * (1.0 + np.tanh(z))


@dataclass(frozen=True)
class Blob:
    x: float
    y: float
    r: float


def _alpha0_two_circles_eval(x: np.ndarray, y: np.ndarray, *, b1: Blob, b2: Blob, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xx, yy = np.broadcast_arrays(x, y)
    eps = max(float(eps), 1.0e-12)

    def _one_circle(xc: float, yc: float, r: float) -> np.ndarray:
        r = max(float(r), 1.0e-12)
        phi = np.sqrt((xx - float(xc)) ** 2 + (yy - float(yc)) ** 2) - r
        return _smooth_step(-phi / eps)

    a1 = _one_circle(b1.x, b1.y, b1.r)
    a2 = _one_circle(b2.x, b2.y, b2.r)
    return np.clip(a1 + a2, 0.0, 1.0)


def _mark_inactive_fields(dh: DofHandler, *field_names: str) -> None:
    tags = getattr(dh, "dof_tags", None) or {}
    inactive = set(tags.get("inactive", set()))
    for fname in field_names:
        try:
            sl = np.asarray(dh.get_field_slice(fname), dtype=int).ravel()
        except Exception:
            continue
        inactive.update(int(i) for i in sl)
    tags["inactive"] = inactive
    dh.dof_tags = tags


def _scalar_to_nodes(dh: DofHandler, f: Function, *, num_nodes: int) -> np.ndarray:
    out = np.zeros(num_nodes, dtype=float)
    for gdof, lidx in f._g2l.items():
        _field, node_id = dh._dof_to_node_map[gdof]
        if node_id is None:
            continue
        out[int(node_id)] = float(f.nodal_values[lidx])
    return out


def _vector_to_nodes(dh: DofHandler, vf: VectorFunction, *, num_nodes: int) -> np.ndarray:
    out = np.zeros((num_nodes, 2), dtype=float)
    field_names = list(vf.field_names)
    for gdof, lidx in vf._g2l.items():
        field, node_id = dh._dof_to_node_map[gdof]
        if node_id is None or field not in field_names:
            continue
        out[int(node_id), field_names.index(field)] = float(vf.nodal_values[lidx])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Two detached biofilm blobs (single alpha) in a channel, transported via reference map.",
    )
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--ny", type=int, default=16)
    ap.add_argument("--L", type=float, default=4.0)
    ap.add_argument("--H", type=float, default=1.0)
    ap.add_argument("--q", type=int, default=6)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--t-final", type=float, default=0.2)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--newton-tol", type=float, default=1.0e-5)
    ap.add_argument("--max-it", type=int, default=50)
    ap.add_argument("--ls-mode", type=str, default="dealii", choices=("armijo", "dealii"))
    ap.add_argument("--outdir", type=str, default="examples/biofilms/results/channel_two_detached_alpha")
    ap.add_argument("--vtk-every", type=int, default=1)

    # Inflow profile
    ap.add_argument("--Umax", type=float, default=0.3)
    ap.add_argument("--Tramp", type=float, default=0.1)

    # Two-circle alpha0 definition
    ap.add_argument("--eps", type=float, default=0.05, help="Diffuse interface thickness used for alpha0(x).")
    ap.add_argument("--blob1-x", type=float, default=None)
    ap.add_argument("--blob1-y", type=float, default=None)
    ap.add_argument("--blob1-r", type=float, default=None)
    ap.add_argument("--blob2-x", type=float, default=None)
    ap.add_argument("--blob2-y", type=float, default=None)
    ap.add_argument("--blob2-r", type=float, default=None)
    ap.add_argument("--phi-b", type=float, default=0.3, help="Biofilm porosity (phi inside the blobs).")

    # Physical parameters (dimensionless defaults match biofilm_channel_sloughing.py)
    ap.add_argument("--rho-f", type=float, default=1.0)
    ap.add_argument("--mu-f", type=float, default=0.1)
    ap.add_argument("--kappa-inv", type=float, default=10.0)
    ap.add_argument("--mu-s", type=float, default=0.5)
    ap.add_argument("--lambda-s", type=float, default=0.5)
    ap.add_argument(
        "--solid-model",
        type=str,
        default="linear",
        choices=("linear", "neo_hookean", "neo-hookean", "nh"),
        help="Skeleton constitutive model.",
    )
    ap.add_argument("--solid-inertia", action="store_true", default=True)
    ap.add_argument("--no-solid-inertia", dest="solid_inertia", action="store_false")
    ap.add_argument("--rho-s0", type=float, default=1.0, help="Skeleton inertia coefficient (used when --solid-inertia).")

    # u-extension stabilization (critical for multi-component alpha)
    ap.add_argument("--u-extension", type=str, default="grad", choices=("l2", "grad"))
    ap.add_argument("--gamma-u", type=float, default=2.0)
    ap.add_argument("--gamma-u-pin", type=float, default=1.0e-12)
    ap.add_argument("--u-cip", type=float, default=1.0, help="Facet stabilization for u (CIP / ghost penalty).")

    args = ap.parse_args()

    L = float(args.L)
    H = float(args.H)
    qdeg = int(args.q)
    dt_val = float(args.dt)
    theta = float(args.theta)
    backend = str(args.backend)

    os.makedirs(str(args.outdir), exist_ok=True)

    # Default blob geometry if not specified: two equal circles, detached from walls.
    r_default = 0.18 * H
    b1 = Blob(
        x=float(args.blob1_x) if args.blob1_x is not None else 0.35 * L,
        y=float(args.blob1_y) if args.blob1_y is not None else 0.35 * H,
        r=float(args.blob1_r) if args.blob1_r is not None else r_default,
    )
    b2 = Blob(
        x=float(args.blob2_x) if args.blob2_x is not None else 0.65 * L,
        y=float(args.blob2_y) if args.blob2_y is not None else 0.65 * H,
        r=float(args.blob2_r) if args.blob2_r is not None else r_default,
    )
    eps = float(args.eps)
    phi_b = float(args.phi_b)

    # ------------------------------------------------------------------
    # Mesh / FE setup
    # ------------------------------------------------------------------
    nodes, elems, _, corners = structured_quad(L, H, nx=int(args.nx), ny=int(args.ny), poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    _tag_rectangle_boundaries(mesh, L=L, H=H)

    me = MixedElement(
        mesh,
        field_specs={
            "v_x": 2,
            "v_y": 2,
            "p": 1,
            "vS_x": 2,
            "vS_y": 2,
            "u_x": 2,
            "u_y": 2,
            "phi": 1,
            "alpha": 1,
            "S": 1,
        },
    )
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
    VS = FunctionSpace("VS", ["vS_x", "vS_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    dv = VectorTrialFunction(space=V, dof_handler=dh)
    dvS = VectorTrialFunction(space=VS, dof_handler=dh)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    dp = TrialFunction("p", dof_handler=dh)
    dphi = TrialFunction("phi", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    dS = TrialFunction("S", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    vS_test = VectorTestFunction(space=VS, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    S_test = TestFunction("S", dof_handler=dh)

    # Unknowns (k) and previous state (n)
    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    vS_k = VectorFunction("vS_k", ["vS_x", "vS_y"], dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    S_k = Function("S_k", "S", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    vS_n = VectorFunction("vS_n", ["vS_x", "vS_y"], dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    S_n = Function("S_n", "S", dof_handler=dh)

    # Freeze (alpha,phi,S): prescribe via refmap + phi(alpha), keep S=0.
    _mark_inactive_fields(dh, "alpha", "phi", "S")

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------
    alpha_xy = np.asarray(dh.get_dof_coords("alpha"), dtype=float)
    alpha0 = _alpha0_two_circles_eval(alpha_xy[:, 0], alpha_xy[:, 1], b1=b1, b2=b2, eps=eps).ravel()
    alpha_n.nodal_values[:] = alpha0
    alpha_k.nodal_values[:] = alpha0

    # Set phi consistent with alpha (phi=phi_b in biofilm, 1 in free fluid).
    phi_xy = np.asarray(dh.get_dof_coords("phi"), dtype=float)
    if phi_xy.shape == alpha_xy.shape and np.allclose(phi_xy, alpha_xy, rtol=0.0, atol=1.0e-14):
        phi0 = 1.0 - (1.0 - phi_b) * alpha0
    else:
        phi0 = 1.0 - (1.0 - phi_b) * _alpha0_two_circles_eval(phi_xy[:, 0], phi_xy[:, 1], b1=b1, b2=b2, eps=eps).ravel()
    phi_n.nodal_values[:] = np.asarray(phi0, dtype=float)
    phi_k.nodal_values[:] = np.asarray(phi0, dtype=float)

    S_n.set_values_from_function(lambda x, y: 0.0)
    S_k.set_values_from_function(lambda x, y: 0.0)
    v_n.set_values_from_function(lambda x, y: np.array([0.0, 0.0], dtype=float))
    vS_n.set_values_from_function(lambda x, y: np.array([0.0, 0.0], dtype=float))
    u_n.set_values_from_function(lambda x, y: np.array([0.0, 0.0], dtype=float))
    p_n.set_values_from_function(lambda x, y: 0.0)

    # ------------------------------------------------------------------
    # Reference-map transport of alpha + consistent phi(alpha)
    # ------------------------------------------------------------------
    num_nodes = len(mesh.nodes_list)
    alpha_dof_xy = alpha_xy
    alpha_dof_gids = np.asarray(dh.get_field_slice("alpha"), dtype=int).ravel()
    alpha_node_ids = []
    for gd in alpha_dof_gids:
        _fld, nid = dh._dof_to_node_map.get(int(gd), (None, None))
        if nid is None:
            raise RuntimeError("alpha-from-refmap requires CG alpha DOFs to be node-attached.")
        alpha_node_ids.append(int(nid))
    alpha_node_ids = np.asarray(alpha_node_ids, dtype=int)

    phi_dof_xy = phi_xy
    phi_dof_gids = np.asarray(dh.get_field_slice("phi"), dtype=int).ravel()
    phi_node_ids = []
    for gd in phi_dof_gids:
        _fld, nid = dh._dof_to_node_map.get(int(gd), (None, None))
        if nid is None:
            raise RuntimeError("phi update requires CG phi DOFs to be node-attached.")
        phi_node_ids.append(int(nid))
    phi_node_ids = np.asarray(phi_node_ids, dtype=int)

    def _update_alpha_phi_from_refmap() -> None:
        u_nodes = _vector_to_nodes(dh, u_k, num_nodes=num_nodes)

        # Alpha nodes
        chi_a = alpha_dof_xy - u_nodes[alpha_node_ids, :]
        a_vals = _alpha0_two_circles_eval(chi_a[:, 0], chi_a[:, 1], b1=b1, b2=b2, eps=eps).ravel()
        a_vals = np.clip(a_vals, 0.0, 1.0)
        alpha_k.nodal_values[:] = np.asarray(a_vals, dtype=float)

        # Phi nodes: keep phi consistent with alpha on their respective node sets.
        chi_p = phi_dof_xy - u_nodes[phi_node_ids, :]
        a_p = _alpha0_two_circles_eval(chi_p[:, 0], chi_p[:, 1], b1=b1, b2=b2, eps=eps).ravel()
        phi_vals = 1.0 - (1.0 - phi_b) * np.clip(a_p, 0.0, 1.0)
        phi_k.nodal_values[:] = np.asarray(phi_vals, dtype=float)

    # ------------------------------------------------------------------
    # Forms
    # ------------------------------------------------------------------
    dt_c = Constant(dt_val)

    solid_model = str(getattr(args, "solid_model", "linear")).strip().lower()
    if solid_model in {"nh", "neo-hookean", "neo_hookean"}:
        solid_model = "neo_hookean"

    forms = build_biofilm_one_domain_forms(
        v_k=v_k,
        p_k=p_k,
        vS_k=vS_k,
        u_k=u_k,
        phi_k=phi_k,
        alpha_k=alpha_k,
        S_k=S_k,
        v_n=v_n,
        p_n=p_n,
        vS_n=vS_n,
        u_n=u_n,
        phi_n=phi_n,
        alpha_n=alpha_n,
        S_n=S_n,
        dv=dv,
        dp=dp,
        dvS=dvS,
        du=du,
        dphi=dphi,
        dalpha=dalpha,
        dS=dS,
        v_test=v_test,
        q_test=q_test,
        vS_test=vS_test,
        u_test=u_test,
        phi_test=phi_test,
        alpha_test=alpha_test,
        S_test=S_test,
        dx=dx(metadata={"q": int(qdeg)}),
        dt=dt_c,
        theta=float(theta),
        rho_f=Constant(float(args.rho_f)),
        mu_f=Constant(float(args.mu_f)),
        kappa_inv=Constant(float(args.kappa_inv)),
        mu_s=Constant(float(args.mu_s)),
        lambda_s=Constant(float(args.lambda_s)),
        solid_model=solid_model,
        rho_s0_tilde=Constant(float(args.rho_s0)),
        include_skeleton_acceleration=bool(args.solid_inertia),
        gamma_u=float(args.gamma_u),
        u_extension_mode=str(getattr(args, "u_extension", "grad")),
        gamma_u_pin=float(getattr(args, "gamma_u_pin", 0.0)),
        u_cip=float(getattr(args, "u_cip", 0.0)),
        ds_cip=ds(metadata={"q": int(qdeg)}),
        # All transports disabled: alpha/phi are prescribed, S=0.
        D_phi=0.0,
        gamma_phi=0.0,
        D_alpha=0.0,
        D_S=0.0,
        mu_max=0.0,
        k_g=0.0,
        k_d=0.0,
        k_det=0.0,
        f_v=None,
        f_u=None,
        s_v=None,
        f_phi=None,
        f_alpha=None,
        f_S=None,
    )

    # ------------------------------------------------------------------
    # Dirichlet BCs: channel flow
    # ------------------------------------------------------------------
    Umax = float(args.Umax)
    Tramp = max(1.0e-12, float(args.Tramp))

    def ramp(t: float) -> float:
        return 1.0 - float(np.exp(-float(t) / Tramp))

    def inflow_vx(x: float, y: float, t: float) -> float:
        yy = float(y) / H
        return float(Umax * ramp(t) * 4.0 * yy * (1.0 - yy))

    bcs = [
        BoundaryCondition("v_x", "dirichlet", "left", inflow_vx),
        BoundaryCondition("v_y", "dirichlet", "left", lambda x, y, t: 0.0),
        BoundaryCondition("v_x", "dirichlet", "bottom", lambda x, y, t: 0.0),
        BoundaryCondition("v_y", "dirichlet", "bottom", lambda x, y, t: 0.0),
        BoundaryCondition("v_x", "dirichlet", "top", lambda x, y, t: 0.0),
        BoundaryCondition("v_y", "dirichlet", "top", lambda x, y, t: 0.0),
        BoundaryCondition("p", "dirichlet", "right", lambda x, y, t: 0.0),
    ]
    bcs_homog = [BoundaryCondition(b.field, b.method, b.domain_tag, lambda x, y: 0.0) for b in bcs]

    # ------------------------------------------------------------------
    # Output callback (VTK)
    # ------------------------------------------------------------------
    step_counter = {"k": 0}

    def post_step(functions):
        step_counter["k"] += 1
        step_no = int(step_counter["k"])
        if int(args.vtk_every) <= 0 or (step_no % int(args.vtk_every)) != 0:
            return

        num_nodes_local = len(mesh.nodes_list)
        alpha_nodes = _scalar_to_nodes(dh, alpha_k, num_nodes=num_nodes_local)
        phi_nodes = _scalar_to_nodes(dh, phi_k, num_nodes=num_nodes_local)
        u_nodes = _vector_to_nodes(dh, u_k, num_nodes=num_nodes_local)

        export_vtk(
            filename=os.path.join(str(args.outdir), f"solution_{step_no:04d}.vtu"),
            mesh=mesh,
            dof_handler=dh,
            functions={
                "v": v_k,
                "p": p_k,
                "vS": vS_k,
                "u": u_k,
                "alpha": alpha_nodes,
                "phi": phi_nodes,
                "u_nodes": u_nodes,
            },
        )

    # ------------------------------------------------------------------
    # Solve in time (with refmap alpha update after each accepted step)
    # ------------------------------------------------------------------
    solver = NewtonSolver(
        forms.residual_form,
        forms.jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(
            newton_tol=float(args.newton_tol),
            max_newton_iter=int(args.max_it),
            ls_mode=str(getattr(args, "ls_mode", "dealii")),
        ),
        quad_order=qdeg,
        backend=backend,
        postproc_timeloop_cb=post_step,
    )

    def post_step_refiner(step, bcs_now, functions, prev_functions):
        _update_alpha_phi_from_refmap()
        alpha_k.nodal_values[:] = np.clip(alpha_k.nodal_values, 0.0, 1.0)
        phi_k.nodal_values[:] = np.clip(phi_k.nodal_values, 0.0, 1.0)

    solver.solve_time_interval(
        functions=[v_k, p_k, vS_k, u_k, phi_k, alpha_k, S_k],
        prev_functions=[v_n, p_n, vS_n, u_n, phi_n, alpha_n, S_n],
        aux_functions={
            "dt": dt_c,
        },
        time_params=TimeStepperParameters(
            dt=dt_val,
            final_time=float(args.t_final),
            max_steps=10_000,
            theta=theta,
        ),
        post_step_refiner=post_step_refiner,
    )


if __name__ == "__main__":
    main()
