import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant, VectorFunction, VectorTestFunction, VectorTrialFunction, grad, inner
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad

from examples.utils.shared.nonlinear_solid_refmap import (
    dsigma_hencky_miehe_split,
    sigma_hencky_miehe_split,
)


def _build_u_problem(*, q: int = 4):
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )

    me = MixedElement(mesh, field_specs={"u_x": 1, "u_y": 1})
    dh = DofHandler(me, method="cg")

    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    u_test = VectorTestFunction(space=U, dof_handler=dh)
    u = VectorFunction("u", ["u_x", "u_y"], dof_handler=dh)

    # Deterministic small-strain-ish state: keep I-∇u invertible and away from degeneracy.
    u.components[0].set_values_from_function(lambda x, y: float(0.10 * x + 0.02 * y))
    u.components[1].set_values_from_function(lambda x, y: float(0.02 * x - 0.05 * y))

    mu_s = Constant(1.0)
    lambda_s = Constant(1.0)
    g = Constant(0.7)
    eta_pos = 1.0e-12
    disc_reg = 1.0e-16

    dx_q = dx(metadata={"q": int(q)})
    return dh, u, du, u_test, mu_s, lambda_s, g, eta_pos, disc_reg, dx_q


def test_hencky_miehe_split_jacobian_fd_consistency():
    dh, u, du, u_test, mu_s, lambda_s, g, eta_pos, disc_reg, dx_q = _build_u_problem(q=2)

    sig_plus, sig_minus = sigma_hencky_miehe_split(u, mu_s, lambda_s, dim=2, eta_pos=eta_pos, disc_reg=disc_reg)
    dsig_plus, dsig_minus = dsigma_hencky_miehe_split(u, du, mu_s, lambda_s, dim=2, eta_pos=eta_pos, disc_reg=disc_reg)

    r = (g * sig_plus + sig_minus)
    dr = (g * dsig_plus + dsig_minus)
    eq = Equation(inner(dr, grad(u_test)) * dx_q, inner(r, grad(u_test)) * dx_q)

    K, R0 = assemble_form(eq, dof_handler=dh, bcs=[], quad_order=3, backend="python")
    R0 = np.asarray(R0, dtype=float)

    def assemble_residual():
        _, R = assemble_form(
            Equation(None, inner(r, grad(u_test)) * dx_q), dof_handler=dh, bcs=[], quad_order=3, backend="python"
        )
        return np.asarray(R, dtype=float)

    # Probe a few u DOFs.
    probes = []
    for fld in ("u_x", "u_y"):
        sl = dh.get_field_slice(fld)
        if sl:
            probes.append(int(sl[len(sl) // 2]))

    eps = 1.0e-6
    for j in probes:
        fld, _ = dh._dof_to_node_map[j]
        func = u.components[0] if fld == "u_x" else u.components[1]
        old = float(func.get_nodal_values(np.asarray([j], dtype=int))[0])
        func.set_nodal_values(np.asarray([j], dtype=int), np.asarray([old + eps], dtype=float))
        R1 = assemble_residual()
        func.set_nodal_values(np.asarray([j], dtype=int), np.asarray([old], dtype=float))

        fd = (R1 - R0) / eps
        col = np.asarray(K.getcol(j).toarray()).reshape(-1)
        denom = max(1.0, float(np.linalg.norm(fd, ord=np.inf)), float(np.linalg.norm(col, ord=np.inf)))
        rel = float(np.linalg.norm(fd - col, ord=np.inf)) / denom
        assert rel < 5.0e-6, f"hencky miehe FD mismatch at dof {j} ({fld}): rel={rel:.2e}"
