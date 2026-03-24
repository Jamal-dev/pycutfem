import numpy as np
import pytest


def _compiled_backends() -> list[str]:
    out = ["jit"]
    try:
        from pycutfem.jit.cpp_backend import compile_backend_cpp  # noqa: F401
    except Exception:
        return out
    out.append("cpp")
    return out


@pytest.mark.parametrize("backend", _compiled_backends())
def test_composite_scalar_grad_matches_python(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_jit_cache_{backend}"))

    from pycutfem.core.dofhandler import DofHandler
    from pycutfem.core.mesh import Mesh
    from pycutfem.fem.mixedelement import MixedElement
    from pycutfem.ufl.expressions import Constant, Function, TestFunction, TrialFunction, exp, grad, inner
    from pycutfem.ufl.forms import Equation, assemble_form
    from pycutfem.ufl.measures import dx
    from pycutfem.utils.meshgen import structured_quad

    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    me = MixedElement(mesh, field_specs={"z": 1})
    dh = DofHandler(me, method="cg")

    dz = TrialFunction("z", dof_handler=dh)
    wz = TestFunction("z", dof_handler=dh)
    z_k = Function("z_k", "z", dof_handler=dh)

    coords = np.asarray(dh.get_dof_coords("z"), dtype=float)
    z_k.nodal_values[:] = 0.35 - 0.4 * coords[:, 0] + 0.2 * coords[:, 1]

    one = Constant(1.0)
    sig = one / (one + exp(-z_k))
    sig_prime = sig * (one - sig)

    qdeg = 6
    dx_q = dx(metadata={"q": qdeg})
    residual = inner(grad(sig), grad(wz)) * dx_q
    jacobian = inner(grad(sig_prime * dz), grad(wz)) * dx_q
    equation = Equation(jacobian, residual)

    K_py, R_py = assemble_form(equation, dof_handler=dh, bcs=[], quad_order=qdeg, backend="python")
    K_backend, R_backend = assemble_form(equation, dof_handler=dh, bcs=[], quad_order=qdeg, backend=backend)

    A_py = K_py.tocsr().toarray()
    A_backend = K_backend.tocsr().toarray()
    dA = float(np.max(np.abs(A_backend - A_py)))
    dR = float(np.max(np.abs(np.asarray(R_backend, dtype=float) - np.asarray(R_py, dtype=float))))

    assert np.isfinite(dA)
    assert np.isfinite(dR)
    assert dA < 1.0e-9
    assert dR < 1.0e-10
