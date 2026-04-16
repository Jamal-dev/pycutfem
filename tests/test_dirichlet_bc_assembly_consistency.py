import numpy as np
import scipy.sparse as sp
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant, TestFunction, TrialFunction, grad, inner
from pycutfem.ufl.forms import BoundaryCondition, Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad


def _build_scalar_dirichlet_problem():
    nodes, elems, _, corners = structured_quad(
        1.0,
        1.0,
        nx=2,
        ny=2,
        poly_order=1,
        offset=(0.0, 0.0),
    )
    mesh = Mesh(
        nodes,
        elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    boundary_tags = {
        "boundary": lambda x, y: (
            np.isclose(x, 0.0)
            | np.isclose(x, 1.0)
            | np.isclose(y, 0.0)
            | np.isclose(y, 1.0)
        )
    }
    dh.tag_dofs_by_locator_map(boundary_tags, fields=["u"])

    u = TrialFunction("u", dof_handler=dh)
    v = TestFunction("u", dof_handler=dh)
    equation = Equation(inner(grad(u), grad(v)) * dx, Constant(1.0) * v * dx)
    bcs = [BoundaryCondition("u", "dirichlet", "boundary", 0.0)]
    return dh, equation, bcs


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_assemble_form_dirichlet_application_matches_reduction(monkeypatch, backend: str) -> None:
    if backend == "cpp":
        monkeypatch.setenv("PYCUTFEM_CPP_FAST_COMPILE", "1")

    dh, equation, bcs = _build_scalar_dirichlet_problem()

    K_raw, F_raw = assemble_form(equation, dof_handler=dh, bcs=[], backend=backend)
    K_bc, F_bc = assemble_form(equation, dof_handler=dh, bcs=bcs, backend=backend)

    K_red_ref, F_red_ref, free_ref, *_ = dh.reduce_linear_system(
        K_raw, F_raw, bcs=bcs, return_dirichlet=True
    )
    K_red_bc, F_red_bc, free_bc, *_ = dh.reduce_linear_system(
        K_bc, F_bc, bcs=bcs, return_dirichlet=True
    )

    assert sp.isspmatrix_csr(K_bc)
    np.testing.assert_array_equal(free_bc, free_ref)
    np.testing.assert_allclose(K_red_bc.toarray(), K_red_ref.toarray(), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(F_red_bc, F_red_ref, rtol=1.0e-12, atol=1.0e-12)
