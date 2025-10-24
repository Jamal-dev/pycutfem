from types import MethodType
from pathlib import Path

import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import (
    Constant,
    Expression,
    VectorFunction,
    det,
    grad,
    inner,
    inv,
)
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.helpers import GradOpInfo
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad
from pycutfem.jit.cache import KernelCache


def _setup_single_q1_vector():
    nodes, elems, _, corners = structured_quad(
        1,
        1,
        nx=1,
        ny=1,
        poly_order=1,
        parallel=False,
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    mixed_element = MixedElement(mesh, field_specs={"ux": 1, "uy": 1})
    dof_handler = DofHandler(mixed_element, method="cg")
    return mesh, dof_handler


class _FakeGrad(Expression):
    """Minimal expression used to inject a GradOpInfo result into the compiler."""


def test_python_det_inv_materialize_gradinfo():
    _, dof_handler = _setup_single_q1_vector()
    compiler = FormCompiler(dof_handler, backend="python")

    mat = np.array([[1.2, 0.3], [-0.1, 0.9]], dtype=float)

    # Constant tensors go through the numpy path.
    det_expr = det(Constant(mat))
    inv_expr = inv(Constant(mat))
    assert np.isclose(compiler._visit(det_expr), np.linalg.det(mat), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(compiler._visit(inv_expr), np.linalg.inv(mat), rtol=1e-12, atol=1e-12)

    # GradOpInfo (function-valued) path.
    grad_info = GradOpInfo(np.array(mat, dtype=float), role="function")

    # Inject a visitor for our fake node that returns grad_info.
    def _fake_visitor(self, node):
        return grad_info

    compiler._dispatch[_FakeGrad] = MethodType(_fake_visitor, compiler)
    try:
        det_val = compiler._visit(det(_FakeGrad()))
        inv_val = compiler._visit(inv(_FakeGrad()))
    finally:
        compiler._dispatch.pop(_FakeGrad, None)

    assert np.isclose(det_val, np.linalg.det(mat), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(inv_val, np.linalg.inv(mat), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("backend", ["jit"])
def test_det_inv_functional_jit(backend):
    # Redirect kernel cache to the workspace to avoid permission errors.
    cache_dir = Path(".pytest_cache") / "pycutfem_jit"
    cache_dir.mkdir(parents=True, exist_ok=True)
    KernelCache._CACHE_DIR = cache_dir

    mesh, dof_handler = _setup_single_q1_vector()
    a, b = 0.1, -0.05
    disp = VectorFunction("disp", ["ux", "uy"], dof_handler=dof_handler)
    disp.set_values_from_function(lambda x, y: np.array([a * x, b * y]))

    identity = Constant(np.eye(2))
    F = identity + grad(disp)

    # det(F) functional
    det_integral = det(F) * dx(metadata={"q": 3})
    det_hooks = {type(det_integral.integrand): {"name": "detF"}}
    det_res = assemble_form(
        Equation(None, det_integral),
        dof_handler=dof_handler,
        bcs=[],
        assembler_hooks=det_hooks,
        backend=backend,
    )
    area = mesh.areas()[0]
    expected_det = (1.0 + a) * (1.0 + b) * area
    assert np.isclose(det_res["detF"], expected_det, rtol=1e-12, atol=1e-12)

    # det(inv(F)) uses both operations.
    det_inv_integral = det(inv(F)) * dx(metadata={"q": 3})
    det_inv_hooks = {type(det_inv_integral.integrand): {"name": "detFinv"}}
    det_inv_res = assemble_form(
        Equation(None, det_inv_integral),
        dof_handler=dof_handler,
        bcs=[],
        assembler_hooks=det_inv_hooks,
        backend=backend,
    )
    expected_det_inv = area / ((1.0 + a) * (1.0 + b))
    assert np.isclose(det_inv_res["detFinv"], expected_det_inv, rtol=1e-12, atol=1e-12)
