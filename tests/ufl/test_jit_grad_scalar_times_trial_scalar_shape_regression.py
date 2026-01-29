import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.jit import compile_backend
from pycutfem.ufl.expressions import (
    Constant,
    Function,
    TestFunction,
    TrialFunction,
    VectorFunction,
    dot,
    grad,
)
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def test_jit_grad_scalar_times_trial_scalar_adds_to_grad_trial_without_broadcast_error(tmp_path, monkeypatch):
    """
    Regression:
    In the biofilm/FPI Jacobians we get terms like

        grad(trial_alpha) * scalar_value + grad(alpha_value) * trial_phi

    where grad(trial_alpha) is represented as a grad-tensor (1,n,d) and the
    second term must follow the same convention to be addable. Historically the
    JIT codegen produced a (d,n) array for grad(alpha_value) * trial_phi, which
    then failed to broadcast in '+'.
    """
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "jitcache"))

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
            "p": 1,
            "phi": 1,
            "alpha": 1,
        },
    )
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["v_x", "v_y"], dim=1)
    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)

    dphi = TrialFunction("phi", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)

    expr = q_test * dot(grad(dalpha) * (phi_k - Constant(1.0)) + grad(alpha_k) * dphi, v_k)

    runner, _ = compile_backend(expr, dh, me, on_facet=False)
    assert runner is not None

