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
def test_composite_scalar_grad_linearity_and_chain_rule_match_python(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_chain_rule_{backend}"))

    from pycutfem.core.dofhandler import DofHandler
    from pycutfem.core.mesh import Mesh
    from pycutfem.fem.mixedelement import MixedElement
    from pycutfem.ufl.expressions import Constant, Function, TestFunction, exp, grad, inner
    from pycutfem.ufl.forms import Equation, assemble_form
    from pycutfem.ufl.measures import dx
    from pycutfem.utils.meshgen import structured_quad

    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=2)
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

    w = TestFunction("z", dof_handler=dh)
    z_k = Function("z_k", "z", dof_handler=dh)

    coords = np.asarray(dh.get_dof_coords("z"), dtype=float)
    z_k.nodal_values[:] = 0.2 + 0.3 * coords[:, 0] - 0.15 * coords[:, 1]

    one = Constant(1.0)
    coeff = (one + z_k) * exp(z_k)
    expanded_grad = exp(z_k) * grad(z_k) + coeff * grad(z_k)

    dΩ = dx(metadata={"q": 6})
    compact_form = inner(grad(coeff), grad(w)) * dΩ
    expanded_form = inner(expanded_grad, grad(w)) * dΩ

    _, rhs_compact_py = assemble_form(Equation(None, compact_form), dof_handler=dh, bcs=[], backend="python")
    _, rhs_expanded_py = assemble_form(Equation(None, expanded_form), dof_handler=dh, bcs=[], backend="python")
    _, rhs_compact_backend = assemble_form(Equation(None, compact_form), dof_handler=dh, bcs=[], backend=backend)
    _, rhs_expanded_backend = assemble_form(Equation(None, expanded_form), dof_handler=dh, bcs=[], backend=backend)

    rhs_compact_py = np.asarray(rhs_compact_py, dtype=float).reshape(-1)
    rhs_expanded_py = np.asarray(rhs_expanded_py, dtype=float).reshape(-1)
    rhs_compact_backend = np.asarray(rhs_compact_backend, dtype=float).reshape(-1)
    rhs_expanded_backend = np.asarray(rhs_expanded_backend, dtype=float).reshape(-1)

    np.testing.assert_allclose(rhs_expanded_py, rhs_compact_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(rhs_compact_backend, rhs_compact_py, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(rhs_expanded_backend, rhs_compact_py, rtol=1.0e-10, atol=1.0e-10)
