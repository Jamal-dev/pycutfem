import argparse

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
)
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.ufl.measures import dx
from examples.utils.biofilm.one_domain import build_biofilm_one_domain_forms
from pycutfem.utils.meshgen import structured_quad


def _build_problem(*, nx: int, ny: int, q: int, seed: int, dt_val: float, theta: float, solid_model: str, kappa_inv_model: str, rho_s0: float):
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )

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

    rng = np.random.default_rng(int(seed))
    for vf in (v_k, vS_k, u_k, v_n, vS_n, u_n):
        vf.nodal_values[:] = 1.0e-2 * rng.standard_normal(vf.nodal_values.shape)
    for sf in (p_k, p_n):
        sf.nodal_values[:] = 1.0e-2 * rng.standard_normal(sf.nodal_values.shape)

    # Keep nonlinear scalars in safe ranges (avoid K_S + S ~= 0).
    phi_k.nodal_values[:] = np.clip(0.7 + 0.05 * rng.standard_normal(phi_k.nodal_values.shape), 0.2, 0.95)
    phi_n.nodal_values[:] = np.clip(0.7 + 0.05 * rng.standard_normal(phi_n.nodal_values.shape), 0.2, 0.95)
    alpha_k.nodal_values[:] = np.clip(0.5 + 0.05 * rng.standard_normal(alpha_k.nodal_values.shape), 0.05, 0.95)
    alpha_n.nodal_values[:] = np.clip(0.5 + 0.05 * rng.standard_normal(alpha_n.nodal_values.shape), 0.05, 0.95)
    S_k.nodal_values[:] = np.clip(0.2 + 0.05 * rng.standard_normal(S_k.nodal_values.shape), 0.01, 1.0)
    S_n.nodal_values[:] = np.clip(0.2 + 0.05 * rng.standard_normal(S_n.nodal_values.shape), 0.01, 1.0)

    kappa_inv_model = str(kappa_inv_model).strip().lower()
    if kappa_inv_model == "refmap":
        kappa_inv = Constant(10.0 * np.eye(2, dtype=float))
    else:
        kappa_inv = Constant(10.0)

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
        dx=dx(metadata={"q": int(q)}),
        dt=Constant(float(dt_val)),
        theta=float(theta),
        rho_f=Constant(1.0),
        mu_f=Constant(1.0e-2),
        kappa_inv=kappa_inv,
        kappa_inv_model=kappa_inv_model,
        solid_model=str(solid_model).strip().lower(),
        rho_s0_tilde=Constant(float(rho_s0)),
        include_skeleton_acceleration=bool(float(rho_s0) != 0.0),
        mu_s=Constant(1.0),
        lambda_s=Constant(1.0),
        D_phi=0.1,
        gamma_phi=1.0,
        D_alpha=0.1,
        D_S=0.1,
        mu_max=0.4,
        K_S=0.3,
        k_g=0.5,
        k_d=0.1,
        Y=0.8,
        k_det=0.2,
    )

    return dh, forms


def main():
    ap = argparse.ArgumentParser(description="Smoke assembly for one-domain biofilm model (python/jit/cpp backends).")
    ap.add_argument("--nx", type=int, default=2)
    ap.add_argument("--ny", type=int, default=2)
    ap.add_argument("--q", type=int, default=5, help="Quadrature order (passed to dx metadata + assembler).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--solid-model", type=str, default="linear", choices=("linear", "neo_hookean"))
    ap.add_argument("--kappa-inv-model", type=str, default="spatial", choices=("spatial", "refmap"))
    ap.add_argument("--rho-s0", type=float, default=0.0, help="Skeleton inertia coefficient (rho_s0_tilde).")
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--compare-python", action="store_true", help="Also assemble with backend='python' and compare.")
    args = ap.parse_args()

    dh, forms = _build_problem(
        nx=args.nx,
        ny=args.ny,
        q=args.q,
        seed=args.seed,
        dt_val=args.dt,
        theta=args.theta,
        solid_model=args.solid_model,
        kappa_inv_model=args.kappa_inv_model,
        rho_s0=args.rho_s0,
    )
    eq = Equation(forms.jacobian_form, forms.residual_form)

    K, R = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=int(args.q), backend=args.backend)
    R = np.asarray(R, dtype=float)
    print(f"[{args.backend}] |R|_inf = {np.linalg.norm(R, ord=np.inf):.3e}, nnz(K)={K.nnz}")

    if args.compare_python and args.backend != "python":
        K_py, R_py = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=int(args.q), backend="python")
        A = K.tocsr().toarray()
        A_py = K_py.tocsr().toarray()
        dA = float(np.max(np.abs(A - A_py)))
        dR = float(np.max(np.abs(R - np.asarray(R_py, dtype=float))))
        print(f"[compare] max|A-{args.backend}-python| = {dA:.3e}")
        print(f"[compare] max|R-{args.backend}-python| = {dR:.3e}")


if __name__ == "__main__":
    main()
