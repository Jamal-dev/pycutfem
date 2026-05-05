from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import LinearSolverParameters, NewtonParameters, NewtonSolver
from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction, grad, inner
from pycutfem.ufl.forms import BoundaryCondition
from pycutfem.ufl.measures import dx
from pycutfem.utils.gmsh_loader import mesh_from_gmsh


def _write_csv(path: Path, coords: np.ndarray, values: np.ndarray, *, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([coords[:, 0], coords[:, 1], values])
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def main() -> None:
    ap = argparse.ArgumentParser(description="PyCutFEM Poisson parity run on a shared Gmsh mesh.")
    ap.add_argument(
        "--mesh-file",
        type=str,
        default="examples/fracture/mode_I_crack/mesh/miehe_unit_square_32x32.msh",
        help="Gmsh .msh file (must contain physical-line tags: left/right/bottom/top).",
    )
    ap.add_argument("--backend", type=str, default="cpp", choices=("python", "jit", "cpp"))
    ap.add_argument("--q", type=int, default=4, help="Quadrature order.")
    ap.add_argument("--csv", type=str, default="examples/debug/out/parity/poisson_pycutfem.csv")
    args = ap.parse_args()

    mesh = mesh_from_gmsh(str(args.mesh_file), apply_boundary_tags=True)
    order = int(mesh.poly_order)

    me = MixedElement(mesh, field_specs={"u": order})
    dh = DofHandler(me, method="cg")

    u_k = Function("u_k", "u", dof_handler=dh)
    u_n = Function("u_n", "u", dof_handler=dh)
    du = TrialFunction("u", dof_handler=dh)
    u_test = TestFunction("u", dof_handler=dh)

    qdeg = int(args.q)
    dx_form = dx(metadata={"q": qdeg})

    # Parity PDE: -Δu = 1 on Ω, u=0 on ∂Ω.
    f = Constant(1.0)
    residual = inner(grad(u_k), grad(u_test)) * dx_form - f * u_test * dx_form
    jacobian = inner(grad(du), grad(u_test)) * dx_form

    bcs = [
        BoundaryCondition("u", "dirichlet", "left", lambda x, y, t=0.0: 0.0),
        BoundaryCondition("u", "dirichlet", "right", lambda x, y, t=0.0: 0.0),
        BoundaryCondition("u", "dirichlet", "bottom", lambda x, y, t=0.0: 0.0),
        BoundaryCondition("u", "dirichlet", "top", lambda x, y, t=0.0: 0.0),
    ]
    bcs_homog = bcs

    solver = NewtonSolver(
        residual,
        jacobian,
        dof_handler=dh,
        mixed_element=me,
        bcs=bcs,
        bcs_homog=bcs_homog,
        newton_params=NewtonParameters(newton_tol=1.0e-12, max_newton_iter=10, ls_mode="dealii"),
        lin_params=LinearSolverParameters(backend="scipy"),
        quad_order=qdeg,
        backend=str(args.backend),
    )

    u_k.nodal_values.fill(0.0)
    u_n.nodal_values.fill(0.0)

    _delta, converged, _nits = solver._newton_loop([u_k], [u_n], {}, bcs)
    if not converged:
        raise RuntimeError("Poisson Newton did not converge (should converge in 1 step).")

    coords = np.asarray(dh.get_dof_coords("u"), dtype=float)
    values = np.asarray(u_k.nodal_values, dtype=float)
    out_csv = Path(str(args.csv))
    _write_csv(out_csv, coords, values, header="x,y,u")
    print(f"[poisson_pycutfem] dofs={len(values)} u[max]={float(values.max()):.12e} wrote {out_csv}")


if __name__ == "__main__":
    main()
