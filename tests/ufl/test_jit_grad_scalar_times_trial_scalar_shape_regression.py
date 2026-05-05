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
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.ufl.spaces import FunctionSpace
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


def test_python_backend_composite_grad_scalar_times_trial_scalar_assembles():
    """
    Regression for the Python compiler path:
    assembling a composite scalar gradient like

        grad(trial_alpha * scalar_function + alpha_value * trial_phi)

    must not fall through the leaf-only grad visitor and crash on missing fields.
    """
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

    v_k.components[0].nodal_values[:] = 0.1
    v_k.components[1].nodal_values[:] = -0.05
    phi_k.nodal_values[:] = 0.7
    alpha_k.nodal_values[:] = 0.4

    dphi = TrialFunction("phi", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)

    expr = q_test * dot(grad(dalpha * (phi_k - Constant(1.0)) + alpha_k * dphi), v_k)
    form = expr * dx(metadata={"q": 4})

    K_py, F_py = assemble_form(Equation(form, None), dof_handler=dh, bcs=[], quad_order=4, backend="python")
    assert K_py.shape == (dh.total_dofs, dh.total_dofs)
    assert F_py.shape == (dh.total_dofs,)
