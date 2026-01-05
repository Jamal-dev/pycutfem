import numpy as np
import scipy.sparse.linalg as spla
import sympy as sp

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.analytic import Analytic, x, y
from pycutfem.ufl.expressions import (
    TrialFunction,
    TestFunction,
    Constant,
    CellDiameter,
    Laplacian,
    Jump,
    Pos,
    Neg,
    FacetNormal,
    grad,
    inner,
    dot,
)
from pycutfem.ufl.forms import BoundaryCondition, assemble_form
from pycutfem.ufl.measures import dx, ds, dS
from pycutfem.utils.meshgen import structured_quad


def solve_advection_diffusion_supg(
    *,
    nx=16,
    ny=16,
    poly_order=1,
    epsilon=1.0e-2,
    beta_vec=(1.0, 1.0),
    quad_order=5,
    backend="jit",
    method="cg",
):
    beta_vec = np.asarray(beta_vec, dtype=float)
    beta_mag = float(np.linalg.norm(beta_vec))
    if beta_mag <= 0.0:
        raise ValueError("beta_vec must be non-zero for SUPG.")

    u_exact_sym = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    f_sym = (
        -epsilon * sp.diff(u_exact_sym, x, 2)
        - epsilon * sp.diff(u_exact_sym, y, 2)
        + beta_vec[0] * sp.diff(u_exact_sym, x)
        + beta_vec[1] * sp.diff(u_exact_sym, y)
    )
    u_exact_func = sp.lambdify((x, y), u_exact_sym, "numpy")
    u_exact = Analytic(u_exact_sym, degree=4, dim=0)
    f = Analytic(f_sym, degree=4, dim=0)

    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=poly_order)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )

    me = MixedElement(mesh, field_specs={"u": poly_order})
    dh = DofHandler(me, method=method)

    u = TrialFunction("u", dof_handler=dh)
    v = TestFunction("u", dof_handler=dh)
    beta = Constant(beta_vec, dim=1)
    h = CellDiameter()
    tau = h / (2.0 * beta_mag)
    q_meta = {"q": quad_order}

    if method == "cg":
        a = (epsilon * inner(grad(u), grad(v)) + dot(beta, grad(u)) * v) * dx(metadata=q_meta)
        res_u = dot(beta, grad(u)) - epsilon * Laplacian(u)
        a_supg = tau * dot(beta, grad(v)) * res_u * dx(metadata=q_meta)
        L = f * v * dx(metadata=q_meta)
        L_supg = tau * dot(beta, grad(v)) * f * dx(metadata=q_meta)

        bc_tags = {
            "bottom": lambda x, y: np.isclose(y, 0.0),
            "left": lambda x, y: np.isclose(x, 0.0),
            "right": lambda x, y: np.isclose(x, 1.0),
            "top": lambda x, y: np.isclose(y, 1.0),
        }
        mesh.tag_boundary_edges(bc_tags)
        bcs = [
            BoundaryCondition("u", "dirichlet", tag, u_exact_func)
            for tag in bc_tags.keys()
        ]

        K, F = assemble_form(
            (a + a_supg) == (L + L_supg),
            dof_handler=dh,
            bcs=bcs,
            quad_order=quad_order,
            backend=backend,
        )
    else:
        mesh.tag_boundary_edges({"boundary": lambda x, y: True})
        boundary_bs = mesh.edge_bitset("boundary")
        dS_bnd = dS(defined_on=boundary_bs, metadata=q_meta)
        n = FacetNormal()
        penalty = Constant(10.0 * (poly_order + 1) ** 2)
        beta_n = dot(beta, n)
        abs_beta_n = (beta_n * beta_n) ** Constant(0.5)

        a_vol = (epsilon * inner(grad(u), grad(v)) - dot(beta, grad(v)) * u) * dx(metadata=q_meta)
        L = f * v * dx(metadata=q_meta)

        avg_grad_u = 0.5 * (Pos(grad(u)) + Neg(grad(u)))
        avg_grad_v = 0.5 * (Pos(grad(v)) + Neg(grad(v)))
        jump_u = Jump(u)
        jump_v = Jump(v)
        avg_u = 0.5 * (Pos(u) + Neg(u))

        a_int = (
            -epsilon * dot(avg_grad_u, n) * jump_v
            -epsilon * dot(avg_grad_v, n) * jump_u
            + penalty * epsilon / h * jump_u * jump_v
            + beta_n * avg_u * jump_v
            + 0.5 * abs_beta_n * jump_u * jump_v
        ) * ds(metadata=q_meta)

        a_bnd = (
            -epsilon * dot(grad(u), n) * v
            -epsilon * dot(grad(v), n) * u
            + penalty * epsilon / h * u * v
            + 0.5 * (beta_n + abs_beta_n) * u * v
        ) * dS_bnd

        bnd_rhs = (
            -epsilon * dot(grad(v), n)
            + penalty * epsilon / h * v
            + 0.5 * (beta_n - abs_beta_n) * v
        )
        L_bnd = (u_exact * bnd_rhs) * dS_bnd

        K, F = assemble_form(
            (a_vol + a_int + a_bnd) == (L + L_bnd),
            dof_handler=dh,
            bcs=[],
            quad_order=quad_order,
            backend=backend,
        )
    uh = spla.spsolve(K, F)

    coords = dh.get_dof_coords("u")
    exact_vals = u_exact_func(coords[:, 0], coords[:, 1])
    err = np.sqrt(np.mean((uh - exact_vals) ** 2))
    return float(err)


def main():
    err = solve_advection_diffusion_supg(nx=16, ny=16, poly_order=1, epsilon=1.0e-2)
    print(f"SUPG advection-diffusion RMS nodal error = {err:.3e}")


if __name__ == "__main__":
    main()
