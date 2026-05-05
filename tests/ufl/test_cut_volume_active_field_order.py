import numpy as np
import pytest

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.expressions import TrialFunction, TestFunction, inner, grad
from pycutfem.ufl.measures import dx
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.compilers import FormCompiler


BACKENDS = ("jit", "cpp")


def _build_cut_mesh():
    nodes, elements, edges, corners = structured_quad(
        Lx=1.0,
        Ly=1.0,
        nx=4,
        ny=4,
        poly_order=1,
    )
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elements,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    level_set = AffineLevelSet(a=1.0, b=0.0, c=-0.2)
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    return mesh, level_set


@pytest.mark.parametrize("backend", BACKENDS)
def test_cut_volume_active_field_order(backend, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", "")
    mesh, level_set = _build_cut_mesh()
    cut = mesh.element_bitset("cut")
    if cut.cardinality() == 0:
        pytest.skip("No cut elements available for active-field ordering test.")

    me = MixedElement(mesh, field_specs={"a": 1, "b": 1})
    dh = DofHandler(me, method="cg")

    u_b = TrialFunction(name="u_b", field_name="b", dof_handler=dh)
    v_a = TestFunction(name="v_a", field_name="a", dof_handler=dh)
    dxc = dx(
        defined_on=cut,
        level_set=level_set,
        metadata={"side": "+"},
    )
    form = inner(grad(u_b), grad(v_a)) * dxc
    eq = Equation(form, None)

    k_py, _ = FormCompiler(dh, backend="python").assemble(eq, bcs=[])
    k_b, _ = FormCompiler(dh, backend=backend).assemble(eq, bcs=[])

    k_py = k_py.tocsr()
    k_b = k_b.tocsr()

    diff = k_b - k_py
    err = float(np.max(np.abs(diff.data))) if diff.data.size else 0.0
    assert np.max(np.abs(k_py.data)) > 0.0
    assert err < 1.0e-9
