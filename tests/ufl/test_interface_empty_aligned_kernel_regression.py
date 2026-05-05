import numpy as np
import pytest

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.jit import compile_multi
from pycutfem.ufl.expressions import Neg, Pos, TestFunction, TrialFunction
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.measures import dInterface


@pytest.mark.parametrize("jit_backend", ["cpp", "numba"])
def test_compile_multi_interface_handles_empty_aligned_edge_set(monkeypatch, jit_backend: str) -> None:
    """
    Regression:
    When the level set is *not* aligned with mesh facets, the aligned-interface
    edge set is empty. `compile_multi` must still return a complete static dict
    for that (empty) kernel so that sided r** tables / pos_map / neg_map lookups
    do not raise KeyError during compilation.
    """
    monkeypatch.setenv("PYCUTFEM_JIT_BACKEND", str(jit_backend))

    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=3, ny=2, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )

    # Non-aligned interface: x = 0.33 does not coincide with grid lines for nx=3.
    ls = AffineLevelSet(a=1.0, b=0.0, c=-0.33)
    mesh.classify_elements(ls)
    mesh.classify_edges(ls)
    mesh.build_interface_segments(ls)

    assert mesh.edge_bitset("interface").cardinality() == 0, "Expected no aligned interface edges for this setup."

    me = MixedElement(mesh, field_specs={"p": 1})
    dh = DofHandler(me, method="cg")

    u = TrialFunction(name="u", field_name="p", dof_handler=dh)
    v = TestFunction(name="v", field_name="p", dof_handler=dh)

    qdeg = 4
    a = (Pos(u) * Pos(v) + Neg(u) * Neg(v)) * dInterface(level_set=ls, metadata={"q": qdeg})

    kernels = compile_multi(Equation(a, None), dof_handler=dh, mixed_element=me, backend="jit")
    assert kernels, "Expected at least one kernel (interface segments)."

