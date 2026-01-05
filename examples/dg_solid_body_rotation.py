import math
import numpy as np
import scipy.sparse.linalg as spla

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.expressions import (
    TrialFunction,
    TestFunction,
    Constant,
    FacetNormal,
    Jump,
    Pos,
    Neg,
    grad,
    dot,
)
from pycutfem.ufl.forms import assemble_form
from pycutfem.ufl.measures import dx, ds, dS
from pycutfem.utils.meshgen import structured_quad


def _rotation_gaussian(x, y, t, *, x0=0.5, y0=0.0, sigma=0.1):
    theta = 2.0 * math.pi * t
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    x_rot = cos_t * x + sin_t * y
    y_rot = -sin_t * x + cos_t * y
    r2 = (x_rot - x0) ** 2 + (y_rot - y0) ** 2
    return np.exp(-r2 / (2.0 * sigma ** 2))


def _rk4_step(u_vec, dt, rhs):
    k1 = rhs(u_vec)
    k2 = rhs(u_vec + 0.5 * dt * k1)
    k3 = rhs(u_vec + 0.5 * dt * k2)
    k4 = rhs(u_vec + dt * k3)
    return u_vec + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_solid_body_rotation(
    *,
    nx=12,
    ny=12,
    poly_order=1,
    cfl=0.4,
    T=1.0,
    quad_order=4,
    backend="python",
    time_integrator="rk4",
):
    if time_integrator not in {"rk4", "euler"}:
        raise ValueError(f"Unknown time_integrator '{time_integrator}'.")

    nodes, elems, _, corners = structured_quad(
        2.0, 2.0, nx=nx, ny=ny, poly_order=poly_order, offset=(-1.0, -1.0)
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )

    me = MixedElement(mesh, field_specs={"u": poly_order})
    dh = DofHandler(me, method="dg")

    u = TrialFunction("u", dof_handler=dh)
    v = TestFunction("u", dof_handler=dh)

    def vel_func(x, y):
        return np.stack([-2.0 * math.pi * y, 2.0 * math.pi * x], axis=-1)

    beta = Analytic(vel_func, degree=1, dim=1)
    n = FacetNormal()
    beta_n = dot(beta, n)
    abs_beta_n = (beta_n * beta_n) ** Constant(0.5)

    avg_u = 0.5 * (Pos(u) + Neg(u))
    jump_u = Jump(u)
    jump_v = Jump(v)

    q_meta = {"q": quad_order}
    a_mass = (u * v) * dx(metadata=q_meta)
    a_vol = (-dot(beta, grad(v)) * u) * dx(metadata=q_meta)
    a_int = (-beta_n * avg_u * jump_v + 0.5 * abs_beta_n * jump_u * jump_v) * ds(metadata=q_meta)

    mesh.tag_boundary_edges({"boundary": lambda x, y: True})
    boundary_bs = mesh.edge_bitset("boundary")
    a_bnd = (0.5 * (beta_n + abs_beta_n) * u * v) * dS(defined_on=boundary_bs, metadata=q_meta)

    M, _ = assemble_form(a_mass == None, dof_handler=dh, bcs=[], quad_order=quad_order, backend=backend)
    K, _ = assemble_form((a_vol + a_int + a_bnd) == None, dof_handler=dh, bcs=[], quad_order=quad_order, backend=backend)

    gdofs = np.asarray(dh.get_field_slice("u"), dtype=int)
    coords = dh.get_dof_coords("u")
    u_vec = np.zeros(dh.total_dofs, dtype=float)
    u_vec[gdofs] = _rotation_gaussian(coords[:, 0], coords[:, 1], 0.0)

    node_xy = np.asarray(mesh.nodes_x_y_pos, dtype=float)
    r_max = float(np.max(np.hypot(node_xy[:, 0], node_xy[:, 1])))
    c_max = 2.0 * math.pi * r_max
    h = max(mesh.element_char_length(eid) for eid in range(len(mesh.elements_list)))
    dt = cfl * h / ((2 * poly_order + 1) * c_max)
    num_steps = max(1, int(math.ceil(T / dt)))
    dt = T / num_steps

    M_solver = spla.factorized(M.tocsc())

    def rhs(vec):
        return -M_solver(K @ vec)

    for _ in range(num_steps):
        if time_integrator == "rk4":
            u_vec = _rk4_step(u_vec, dt, rhs)
        else:
            u_vec = u_vec + dt * rhs(u_vec)

    exact = {"u": lambda x, y: _rotation_gaussian(x, y, T)}
    err = dh.l2_error(u_vec, exact, quad_order=quad_order, relative=False)
    return float(err)


def convergence_study(
    *,
    base_n=6,
    levels=3,
    poly_order=1,
    cfl=0.4,
    quad_order=4,
    backend="python",
):
    ns = [int(base_n * (2 ** i)) for i in range(levels)]
    errors = []
    for n in ns:
        err = solve_solid_body_rotation(
            nx=n,
            ny=n,
            poly_order=poly_order,
            cfl=cfl,
            quad_order=quad_order,
            backend=backend,
        )
        errors.append(err)

    ooa = []
    for i in range(1, len(errors)):
        if errors[i] > 0.0 and errors[i - 1] > 0.0:
            ooa.append(math.log(errors[i - 1] / errors[i], 2.0))
        else:
            ooa.append(float("nan"))
    return {"ns": ns, "errors": errors, "ooa": ooa}


def main():
    result = convergence_study(base_n=6, levels=3, poly_order=1, cfl=0.4, quad_order=4, backend="python")
    print("DG solid-body rotation convergence:")
    for n, err in zip(result["ns"], result["errors"]):
        print(f"  n={n:>3d}  L2 error={err:.6e}")
    for i, ooa in enumerate(result["ooa"], start=1):
        print(f"  OOA[{result['ns'][i-1]}->{result['ns'][i]}] = {ooa:.3f}")


if __name__ == "__main__":
    main()
