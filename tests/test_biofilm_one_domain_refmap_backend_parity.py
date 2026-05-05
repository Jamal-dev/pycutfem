import os

import numpy as np
import pytest

from examples.utils.shared.nonlinear_solid_refmap import deulerian_k_inv, eulerian_k_inv
from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant, Identity, Function, VectorFunction, VectorTestFunction, VectorTrialFunction
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def _compiled_backends() -> list[str]:
    spec = (os.environ.get("PYCUTFEM_TEST_BACKENDS") or "cpp").strip().lower()
    if not spec:
        return ["cpp"]
    if spec == "all":
        return ["jit", "cpp"]
    backends = [item.strip() for item in spec.split(",") if item.strip() in {"jit", "cpp"}]
    return backends or ["cpp"]


def _dot_components(vec_expr, vec_test, *, dim: int = 2):
    acc = Constant(0.0)
    for i in range(int(dim)):
        acc += vec_expr[i] * vec_test[i]
    return acc


def _build_drag_forms(*, q: int = 4):
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=2)
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
            "vS_x": 2,
            "vS_y": 2,
            "u_x": 2,
            "u_y": 2,
            "phi": 1,
            "alpha": 1,
        },
    )
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
    VS = FunctionSpace("VS", ["vS_x", "vS_y"], dim=1)
    U = FunctionSpace("U", ["u_x", "u_y"], dim=1)

    dv = VectorTrialFunction(space=V, dof_handler=dh)
    dvS = VectorTrialFunction(space=VS, dof_handler=dh)
    du = VectorTrialFunction(space=U, dof_handler=dh)
    w = VectorTestFunction(space=V, dof_handler=dh)
    eta_vS = VectorTestFunction(space=VS, dof_handler=dh)

    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    vS_k = VectorFunction("vS_k", ["vS_x", "vS_y"], dof_handler=dh)
    u_k = VectorFunction("u_k", ["u_x", "u_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)

    rng = np.random.default_rng(0)
    for vf in (v_k, vS_k, u_k):
        vf.nodal_values[:] = 1.0e-2 * rng.standard_normal(vf.nodal_values.shape)
    phi_k.nodal_values[:] = np.clip(0.7 + 0.05 * rng.standard_normal(phi_k.nodal_values.shape), 0.2, 0.95)
    alpha_k.nodal_values[:] = np.clip(0.5 + 0.05 * rng.standard_normal(alpha_k.nodal_values.shape), 0.05, 0.95)

    mu_f = Constant(1.0e-2)
    K_inv_ref = Constant(10.0)
    beta_coeff_k = alpha_k * mu_f * (phi_k * phi_k)
    diff_k = v_k - vS_k
    ddiff = dv - dvS
    K_inv = K_inv_ref * Identity(2)
    k_inv_k = eulerian_k_inv(u_k, K_inv, dim=2)
    dk_inv_k = deulerian_k_inv(u_k, du, K_inv, dim=2)

    kdrag_k = []
    dkdrag_k = []
    for i in range(2):
        comp = Constant(0.0)
        dcomp = Constant(0.0)
        for j in range(2):
            comp += k_inv_k[i, j] * diff_k[j]
            dcomp += k_inv_k[i, j] * ddiff[j] + dk_inv_k[i, j] * diff_k[j]
        kdrag_k.append(comp)
        dkdrag_k.append(dcomp)
    kdrag_k = tuple(kdrag_k)
    dkdrag_k = tuple(dkdrag_k)

    dx_q = dx(metadata={"q": int(q)})
    residual = (
        beta_coeff_k * _dot_components(kdrag_k, w, dim=2)
        - beta_coeff_k * _dot_components(kdrag_k, eta_vS, dim=2)
    ) * dx_q
    jacobian = (
        beta_coeff_k * _dot_components(dkdrag_k, w, dim=2)
        - beta_coeff_k * _dot_components(dkdrag_k, eta_vS, dim=2)
    ) * dx_q

    return dh, Equation(jacobian, residual)


@pytest.mark.parametrize("backend", _compiled_backends())
def test_biofilm_refmap_drag_backend_matches_python(backend):
    dh, equation = _build_drag_forms(q=4)

    K_py, R_py = assemble_form(equation, dof_handler=dh, bcs=[], quad_order=4, backend="python")
    K_backend, R_backend = assemble_form(equation, dof_handler=dh, bcs=[], quad_order=4, backend=backend)

    A_py = K_py.tocsr().toarray()
    A_backend = K_backend.tocsr().toarray()
    dA = float(np.max(np.abs(A_backend - A_py)))
    dR = float(np.max(np.abs(np.asarray(R_backend, dtype=float) - np.asarray(R_py, dtype=float))))

    assert np.isfinite(dA)
    assert np.isfinite(dR)
    assert dA < 1.0e-8
    assert dR < 1.0e-10
