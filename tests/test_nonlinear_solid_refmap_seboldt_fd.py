import numpy as np

from examples.utils.shared.nonlinear_solid_refmap import (
    dsigma_neo_hookean_seboldt,
    sigma_neo_hookean_seboldt,
)
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant, VectorFunction, VectorTestFunction, VectorTrialFunction, grad, inner
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def _build_problem(*, nx: int = 1, ny: int = 1, q: int = 4):
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
            "u_x": 2,
            "u_y": 2,
            "vS_x": 2,
            "vS_y": 2,
        },
    )
    dh = DofHandler(me, method="cg")

    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)
    VS = FunctionSpace("VS", ["vS_x", "vS_y"], dim=1)

    du = VectorTrialFunction(space=U, dof_handler=dh)
    vS_test = VectorTestFunction(space=VS, dof_handler=dh)

    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)

    rng = np.random.default_rng(0)
    u_k.nodal_values[:] = 1.0e-2 * rng.standard_normal(u_k.nodal_values.shape)

    mu_s = Constant(1.0)
    lambda_s = Constant(2.0)
    residual = inner(sigma_neo_hookean_seboldt(u_k, mu_s, lambda_s, dim=2), grad(vS_test)) * dx(metadata={"q": int(q)})
    jacobian = inner(dsigma_neo_hookean_seboldt(u_k, du, mu_s, lambda_s, dim=2), grad(vS_test)) * dx(
        metadata={"q": int(q)}
    )

    field_to_func_k = {
        "u_x": u_k.components[0],
        "u_y": u_k.components[1],
    }
    return dh, jacobian, residual, field_to_func_k


def test_seboldt_refmap_neo_hookean_jacobian_fd_consistency():
    dh, jacobian, residual, field_to_func_k = _build_problem(nx=1, ny=1, q=4)
    eq = Equation(jacobian, residual)
    K, R0 = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=4, backend="python")
    R0 = np.asarray(R0, dtype=float)

    def assemble_residual():
        _, R = assemble_form(Equation(None, residual), dof_handler=dh, bcs=[], quad_order=4, backend="python")
        return np.asarray(R, dtype=float)

    probes = []
    for fld in ("u_x", "u_y"):
        sl = dh.get_field_slice(fld)
        if sl:
            probes.append(int(sl[len(sl) // 2]))

    eps = 1.0e-8
    for j in probes:
        fld, _ = dh._dof_to_node_map[j]
        func = field_to_func_k[fld]
        old = float(func.get_nodal_values(np.asarray([j], dtype=int))[0])
        func.set_nodal_values(np.asarray([j], dtype=int), np.asarray([old + eps], dtype=float))
        R1 = assemble_residual()
        func.set_nodal_values(np.asarray([j], dtype=int), np.asarray([old], dtype=float))

        fd = (R1 - R0) / eps
        col = np.asarray(K.getcol(j).toarray()).reshape(-1)
        denom = max(1.0, float(np.linalg.norm(fd, ord=np.inf)), float(np.linalg.norm(col, ord=np.inf)))
        rel = float(np.linalg.norm(fd - col, ord=np.inf)) / denom
        assert rel < 5.0e-6, f"FD mismatch at dof {j} ({fld}): rel={rel:.2e}"
