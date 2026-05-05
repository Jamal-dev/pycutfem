import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.solvers.nonlinear_solver import NewtonParameters, NewtonSolver
from pycutfem.ufl import (
    Function,
    HdivFunction,
    HdivTestFunction,
    HdivTrialFunction,
    TestFunction,
    TrialFunction,
    div,
    dot,
    dx,
    grad,
    inner,
)
from pycutfem.utils.meshgen import structured_quad
from pycutfem.utils.refinement import TensorRefiner


def _make_single_interface_hanging_mesh() -> Mesh:
    nodes, elems, _edges, corners = structured_quad(1.0, 1.0, nx=2, ny=1, poly_order=1)
    mesh0 = Mesh(
        nodes,
        element_connectivity=elems,
        edges_connectivity=None,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    refiner = TensorRefiner(max_ref=3)
    rx = np.zeros((len(mesh0.elements_list),), dtype=int)
    ry = np.zeros_like(rx)
    ry[0] = 1
    return refiner.refine(mesh0, rx, ry)


def _assemble_reduced_system(backend: str):
    mesh = _make_single_interface_hanging_mesh()
    me = MixedElement(mesh, {"v": ("RT", 0), "p": ("DG", 0), "alpha": 1})
    dh = DofHandler(me, method="cg")

    v_k = HdivFunction(name="v_k", field_name="v", dof_handler=dh)
    p_k = Function(name="p_k", field_name="p", dof_handler=dh)
    alpha_k = Function(name="alpha_k", field_name="alpha", dof_handler=dh)

    dv = HdivTrialFunction("v")
    dp = TrialFunction("p", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    w = HdivTestFunction("v")
    q = TestFunction("p", dof_handler=dh)
    beta = TestFunction("alpha", dof_handler=dh)

    qdeg = 6
    residual_form = (
        alpha_k * inner(v_k, w)
        - p_k * div(w)
        + div(v_k) * q
        + 0.25 * dot(grad(alpha_k), w)
        + (alpha_k - 0.3) * beta
    ) * dx(metadata={"q": qdeg})
    jacobian_form = (
        alpha_k * inner(dv, w)
        - dp * div(w)
        + div(dv) * q
        + 0.25 * dot(grad(dalpha), w)
        + dalpha * beta
    ) * dx(metadata={"q": qdeg})

    solver = NewtonSolver(
        residual_form,
        jacobian_form,
        dof_handler=dh,
        mixed_element=me,
        bcs=[],
        bcs_homog=[],
        newton_params=NewtonParameters(newton_tol=1.0e-10, max_newton_iter=2, line_search=False),
        quad_order=qdeg,
        backend=backend,
    )
    assert solver.constraints is not None, "expected hanging-node constraints on the refined mesh"

    rng = np.random.default_rng(4)
    v_k.nodal_values[:] = rng.standard_normal(v_k.nodal_values.size)
    p_k.nodal_values[:] = rng.standard_normal(p_k.nodal_values.size)
    alpha_coords = np.asarray(dh.get_dof_coords("alpha"), dtype=float)
    alpha_k.nodal_values[:] = 0.45 + 0.10 * alpha_coords[:, 0] - 0.05 * alpha_coords[:, 1]
    solver._enforce_constraints_on_functions([v_k, p_k, alpha_k])

    coeffs = {"v_k": v_k, "p_k": p_k, "alpha_k": alpha_k}
    A_red, R_red = solver._assemble_system_reduced(coeffs, need_matrix=True)
    return A_red.toarray(), np.asarray(R_red, dtype=float)


@pytest.mark.parametrize("backend", ("cpp",))
def test_hdiv_hanging_mixed_backend_parity_matches_python(monkeypatch, tmp_path, backend):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"jit_cache_{backend}"))
    if backend == "cpp":
        monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "cpp")
    else:
        monkeypatch.delenv("PYCUTFEM_JIT_BACKEND", raising=False)

    A_ref, R_ref = _assemble_reduced_system("python")
    A, R = _assemble_reduced_system(backend)

    np.testing.assert_allclose(A, A_ref, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(R, R_ref, rtol=1.0e-10, atol=1.0e-10)
