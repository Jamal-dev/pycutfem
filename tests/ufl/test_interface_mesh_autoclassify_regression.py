import os

import numpy as np
import pytest

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import TestFunction as UflTestFunction, TrialFunction
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dInterface
from pycutfem.utils.meshgen import structured_quad


def _have_cpp_backend() -> bool:
    try:
        from pycutfem.jit.cpp_backend import compile_backend_cpp  # noqa: F401

        return True
    except Exception:
        return False


def _selected_backends(*, default: str = "python") -> list[str]:
    spec = (os.environ.get("PYCUTFEM_TEST_BACKENDS") or os.environ.get("BACKEND") or default).strip()
    if not spec:
        return [default]
    if spec.lower() in {"all", "*"}:
        backends = ["python", "jit", "cpp"]
    else:
        backends = [b.strip() for b in spec.split(",") if b.strip()]
    valid = {"python", "jit", "cpp"}
    unknown = [b for b in backends if b not in valid]
    if unknown:
        raise ValueError(f"Unknown backend(s) {unknown}; valid={sorted(valid)}")
    return backends


@pytest.mark.parametrize("backend", _selected_backends(default="python"))
def test_interface_assembly_autoclassifies_mesh(backend):
    if backend == "cpp" and not _have_cpp_backend():
        pytest.skip("cpp backend unavailable in this environment")

    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=6, ny=6, poly_order=1)
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

    ls0 = AffineLevelSet(a=1.0, b=0.0, c=-0.33)
    ls1 = AffineLevelSet(a=1.0, b=0.0, c=-0.67)

    # Pre-classify to the "wrong" level set first (stale mesh tags/segments).
    mesh.classify_elements(ls0)
    mesh.classify_edges(ls0)
    mesh.build_interface_segments(ls0)

    u = TrialFunction("u", dof_handler=dh)
    v = UflTestFunction("u", dof_handler=dh)
    a = (u * v) * dInterface(level_set=ls1, metadata={"q": 4})

    K_auto, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend=backend)
    assert K_auto is not None
    assert int(getattr(K_auto, "nnz", 0)) > 0

    # Manual classification should now match the auto-classified assembly.
    mesh.classify_elements(ls1)
    mesh.classify_edges(ls1)
    mesh.build_interface_segments(ls1)
    K_manual, _ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend=backend)
    assert K_manual is not None

    np.testing.assert_allclose(K_auto.toarray(), K_manual.toarray(), atol=1.0e-12, rtol=1.0e-12)
