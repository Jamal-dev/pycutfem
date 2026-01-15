import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import MeshSize
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad


def _assemble_scalar(dh: DofHandler, form, backend: str) -> float:
    hook = {type(form.integrand): {"name": "val"}}
    res = assemble_form(
        Equation(None, form),
        dof_handler=dh,
        bcs=[],
        assembler_hooks=hook,
        backend=backend,
    )
    return float(np.asarray(res["val"], dtype=float).reshape(-1)[0])


def test_meshsize_integral_matches_backends_and_expected() -> None:
    """
    Regression:
    `MeshSize()` must evaluate consistently across backends.

    In particular, for quads we mimic NGSolve's `specialcf.mesh_size`:
      h = 2 * sqrt(|detJ|)
    at the quadrature point (for affine quads this is constant).

    On the unit square with one affine Q1 quad:
      detJ = 1/4  =>  h = 1,  ∫_Ω h dx = 1.
    """
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    qdeg = 4
    form = MeshSize() * dx(metadata={"q": qdeg})

    val_py = _assemble_scalar(dh, form, backend="python")
    val_jit = _assemble_scalar(dh, form, backend="jit")

    assert np.isfinite(val_py)
    assert np.isfinite(val_jit)
    assert np.allclose(val_py, val_jit, rtol=1e-11, atol=1e-12)
    assert np.allclose(val_py, 1.0, rtol=1e-11, atol=1e-12)

    try:
        val_cpp = _assemble_scalar(dh, form, backend="cpp")
    except Exception as exc:
        pytest.skip(f"cpp backend unavailable: {exc}")
    assert np.isfinite(val_cpp)
    assert np.allclose(val_cpp, val_jit, rtol=1e-11, atol=1e-12)

