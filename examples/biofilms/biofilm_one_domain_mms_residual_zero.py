import argparse
import os

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.io.vtk import export_vtk
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    VectorTestFunction,
    VectorTrialFunction,
)
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dx
from pycutfem.utils.biofilm_mms_one_domain import build_biofilm_one_domain_mms_affine
from pycutfem.utils.biofilm_one_domain import build_biofilm_one_domain_forms
from pycutfem.utils.meshgen import structured_quad


def _tag_unit_square_boundaries(mesh: Mesh, *, tol: float = 1.0e-12) -> None:
    mesh.tag_boundary_edges(
        {
            "left": lambda x, y: abs(x - 0.0) <= tol,
            "right": lambda x, y: abs(x - 1.0) <= tol,
            "bottom": lambda x, y: abs(y - 0.0) <= tol,
            "top": lambda x, y: abs(y - 1.0) <= tol,
        }
    )


def _set_field(dh: DofHandler, field: str, values: np.ndarray, *, func) -> None:
    gdofs = np.asarray(dh.get_field_slice(field), dtype=int)
    func.set_nodal_values(gdofs, np.asarray(values, dtype=float))


def _residual_inf(dh: DofHandler, res_form, bcs, *, backend: str, quad_order: int) -> float:
    _, F = assemble_form(Equation(None, res_form), dof_handler=dh, bcs=[], quad_order=int(quad_order), backend=backend)
    F = np.asarray(F, dtype=float)
    dirichlet = dh.get_dirichlet_data(bcs) or {}
    bc_rows = np.fromiter(dirichlet.keys(), dtype=int) if dirichlet else np.zeros((0,), dtype=int)
    mask = np.ones(dh.total_dofs, dtype=bool)
    if bc_rows.size:
        mask[bc_rows] = False
    if not np.any(mask):
        return 0.0
    return float(np.linalg.norm(F[mask], ord=np.inf))


def main():
    ap = argparse.ArgumentParser(description="Residual-zero MMS check for the one-domain biofilm model.")
    ap.add_argument("--nx", type=int, default=2)
    ap.add_argument("--ny", type=int, default=2)
    ap.add_argument("--q", type=int, default=6)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--compare-python", action="store_true", help="Also compute the residual with backend='python'.")
    ap.add_argument("--outdir", type=str, default="examples/biofilms/results/mms_residual_zero", help="Directory for writing optional VTK output.")
    ap.add_argument("--vtk-every", type=int, default=0, help="Write VTK snapshot (0 disables).")
    args = ap.parse_args()

    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=int(args.nx), ny=int(args.ny), poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    _tag_unit_square_boundaries(mesh)

    me = MixedElement(
        mesh,
        field_specs={
            "v_x": 2,
            "v_y": 2,
            "p": 1,
            "u_x": 2,
            "u_y": 2,
            "phi": 1,
            "alpha": 1,
            "S": 1,
        },
    )
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    dv = VectorTrialFunction(space=V, dof_handler=dh)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    dp = TrialFunction("p", dof_handler=dh)
    dphi = TrialFunction("phi", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    dS = TrialFunction("S", dof_handler=dh)

    v_test = VectorTestFunction(space=V, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)
    phi_test = TestFunction("phi", dof_handler=dh)
    alpha_test = TestFunction("alpha", dof_handler=dh)
    S_test = TestFunction("S", dof_handler=dh)

    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    p_k = Function("p_k", "p", dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)
    S_k = Function("S_k", "S", dof_handler=dh)

    v_n = VectorFunction("v_n", ["v_x", "v_y"], dof_handler=dh)
    p_n = Function("p_n", "p", dof_handler=dh)
    u_n = VectorFunction("u_n", ["u_x", "u_y"], dof_handler=dh)
    phi_n = Function("phi_n", "phi", dof_handler=dh)
    alpha_n = Function("alpha_n", "alpha", dof_handler=dh)
    S_n = Function("S_n", "S", dof_handler=dh)

    mms = build_biofilm_one_domain_mms_affine(dt_val=float(args.dt))

    for field, fn, vec in (
        ("v_x", mms.v_x, v_k.components[0]),
        ("v_y", mms.v_y, v_k.components[1]),
        ("u_x", mms.u_x, u_k.components[0]),
        ("u_y", mms.u_y, u_k.components[1]),
    ):
        xy = dh.get_dof_coords(field)
        _set_field(dh, field, fn(xy[:, 0], xy[:, 1]), func=vec)

    for fld, fnk, fnn, fk, fn in (
        ("p", mms.p_k, mms.p_n, p_k, p_n),
        ("phi", mms.phi_k, mms.phi_n, phi_k, phi_n),
        ("alpha", mms.alpha_k, mms.alpha_n, alpha_k, alpha_n),
        ("S", mms.S_k, mms.S_n, S_k, S_n),
    ):
        xy = dh.get_dof_coords(fld)
        _set_field(dh, fld, fnk(xy[:, 0], xy[:, 1]), func=fk)
        _set_field(dh, fld, fnn(xy[:, 0], xy[:, 1]), func=fn)

    v_n.nodal_values.fill(0.0)
    u_n.nodal_values.fill(0.0)

    f_v = Analytic(lambda x, y: mms.f_v(x, y), degree=2)
    f_u = Analytic(lambda x, y: mms.f_u(x, y), degree=2)
    s_v = Analytic(lambda x, y: mms.s_v(x, y), degree=2)
    f_phi = Analytic(lambda x, y: mms.f_phi(x, y), degree=2)
    f_alpha = Analytic(lambda x, y: mms.f_alpha(x, y), degree=2)
    f_S = Analytic(lambda x, y: mms.f_S(x, y), degree=2)

    forms = build_biofilm_one_domain_forms(
        v_k=v_k,
        p_k=p_k,
        u_k=u_k,
        phi_k=phi_k,
        alpha_k=alpha_k,
        S_k=S_k,
        v_n=v_n,
        p_n=p_n,
        u_n=u_n,
        phi_n=phi_n,
        alpha_n=alpha_n,
        S_n=S_n,
        dv=dv,
        dp=dp,
        du=du,
        dphi=dphi,
        dalpha=dalpha,
        dS=dS,
        v_test=v_test,
        q_test=q_test,
        u_test=u_test,
        phi_test=phi_test,
        alpha_test=alpha_test,
        S_test=S_test,
        dx=dx(metadata={"q": int(args.q)}),
        dt=Constant(float(args.dt)),
        theta=1.0,
        rho_f=Constant(1.0),
        mu_f=Constant(1.0e-2),
        kappa_inv=Constant(10.0),
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
        f_v=f_v,
        f_u=f_u,
        s_v=s_v,
        f_phi=f_phi,
        f_alpha=f_alpha,
        f_S=f_S,
    )

    def _as_float(fn):
        return lambda x, y: float(np.asarray(fn(np.asarray(x), np.asarray(y))).reshape(()))

    bcs = []
    for tag in ("left", "right", "bottom", "top"):
        bcs.extend(
            [
                BoundaryCondition("v_x", "dirichlet", tag, _as_float(mms.v_x)),
                BoundaryCondition("v_y", "dirichlet", tag, _as_float(mms.v_y)),
                BoundaryCondition("u_x", "dirichlet", tag, _as_float(mms.u_x)),
                BoundaryCondition("u_y", "dirichlet", tag, _as_float(mms.u_y)),
                BoundaryCondition("phi", "dirichlet", tag, _as_float(mms.phi)),
                BoundaryCondition("alpha", "dirichlet", tag, _as_float(mms.alpha)),
                BoundaryCondition("S", "dirichlet", tag, _as_float(mms.S)),
            ]
        )

    res = _residual_inf(dh, forms.residual_form, bcs, backend=args.backend, quad_order=int(args.q))
    print(f"[{args.backend}] |R_free|_inf = {res:.3e}")

    vtk_every = int(getattr(args, "vtk_every", 0) or 0)
    if vtk_every > 0:
        vtk_dir = os.path.join(str(args.outdir), "vtk")
        os.makedirs(vtk_dir, exist_ok=True)
        export_vtk(
            os.path.join(vtk_dir, f"mms_residual_zero_nx={int(args.nx)}_ny={int(args.ny)}.vtu"),
            mesh,
            dh,
            {"v": v_k, "p": p_k, "u": u_k, "phi": phi_k, "alpha": alpha_k, "S": S_k},
        )

    if args.compare_python and args.backend != "python":
        res_py = _residual_inf(dh, forms.residual_form, bcs, backend="python", quad_order=int(args.q))
        print(f"[python] |R_free|_inf = {res_py:.3e}")
        print(f"[compare] abs diff = {abs(res - res_py):.3e}")


if __name__ == "__main__":
    main()
