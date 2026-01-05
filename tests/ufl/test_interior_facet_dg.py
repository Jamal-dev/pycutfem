import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Jump, TestFunction, TrialFunction
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import ds
from pycutfem.utils.meshgen import structured_quad


def _assemble_jump_penalty(backend: str):
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=3, ny=1, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )

    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="dg")

    u = TrialFunction("u", dof_handler=dh)
    v = TestFunction("u", dof_handler=dh)
    a = Jump(u) * Jump(v) * ds(metadata={"q": 4})

    K, _ = assemble_form(
        Equation(a, None),
        dof_handler=dh,
        bcs=[],
        backend=backend,
        quad_order=4,
    )
    return K, dh


def test_interior_facet_jump_python():
    K, dh = _assemble_jump_penalty("python")
    assert K.nnz > 0

    dofs0 = np.asarray(dh.get_elemental_dofs(0), dtype=int)
    dofs1 = np.asarray(dh.get_elemental_dofs(1), dtype=int)
    dofs2 = np.asarray(dh.get_elemental_dofs(2), dtype=int)

    block_02 = K[np.ix_(dofs0, dofs2)].toarray()
    block_01 = K[np.ix_(dofs0, dofs1)].toarray()

    assert np.allclose(block_02, 0.0, atol=1.0e-12)
    assert np.linalg.norm(block_01) > 0.0
    assert np.allclose((K - K.T).toarray(), 0.0, atol=1.0e-12)


def test_interior_facet_jump_jit_matches_python():
    K_py, _ = _assemble_jump_penalty("python")
    K_jit, _ = _assemble_jump_penalty("jit")

    diff = (K_py - K_jit).toarray()
    assert np.allclose(diff, 0.0, atol=1.0e-10)
