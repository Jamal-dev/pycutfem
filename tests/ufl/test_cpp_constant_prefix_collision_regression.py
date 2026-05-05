import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant, TestFunction as ScalarTestFunction
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad


def test_cpp_scalar_constant_named_u_prefix_does_not_get_array_view(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_jit_cache"))

    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    v = ScalarTestFunction("u", dof_handler=dh)
    c = Constant(2.5)
    c._jit_name = "u_supg"
    L = (c * v) * dx(metadata={"q": 4})

    _, F_py = assemble_form(Equation(None, L), dof_handler=dh, bcs=[], backend="python")
    try:
        _, F_cpp = assemble_form(Equation(None, L), dof_handler=dh, bcs=[], backend="cpp")
    except Exception as exc:
        pytest.skip(f"cpp backend unavailable: {exc}")

    assert np.allclose(F_cpp, F_py, rtol=1.0e-12, atol=1.0e-12)
