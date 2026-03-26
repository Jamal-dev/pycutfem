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
def test_composite_scalar_sum_of_promoted_grad_terms_matches_python(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / f"pycutfem_sum_cache_{backend}"))

    from pycutfem.core.dofhandler import DofHandler
    from pycutfem.core.mesh import Mesh
    from pycutfem.fem.mixedelement import MixedElement
    from pycutfem.ufl.expressions import (
        Constant,
        Function,
        TestFunction,
        TrialFunction,
        VectorFunction,
        dot,
        grad,
    )
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
    me = MixedElement(
        mesh,
        field_specs={
            "v_x": 2,
            "v_y": 2,
            "p": 1,
            "phi": 1,
            "alpha": 1,
        },
    )
    dh = DofHandler(me, method="cg")

    v_k = VectorFunction("v_k", ["v_x", "v_y"], dof_handler=dh)
    phi_k = Function("phi_k", "phi", dof_handler=dh)
    alpha_k = Function("alpha_k", "alpha", dof_handler=dh)

    coords_vx = np.asarray(dh.get_dof_coords("v_x"), dtype=float)
    coords_vy = np.asarray(dh.get_dof_coords("v_y"), dtype=float)
    coords_phi = np.asarray(dh.get_dof_coords("phi"), dtype=float)
    coords_alpha = np.asarray(dh.get_dof_coords("alpha"), dtype=float)
    v_k.components[0].nodal_values[:] = 0.2 + 0.3 * coords_vx[:, 0] - 0.1 * coords_vx[:, 1]
    v_k.components[1].nodal_values[:] = -0.15 + 0.05 * coords_vy[:, 0] + 0.25 * coords_vy[:, 1]
    phi_k.nodal_values[:] = 0.7 - 0.1 * coords_phi[:, 0] + 0.05 * coords_phi[:, 1]
    alpha_k.nodal_values[:] = 0.3 + 0.2 * coords_alpha[:, 0] - 0.08 * coords_alpha[:, 1]

    dphi = TrialFunction("phi", dof_handler=dh)
    dalpha = TrialFunction("alpha", dof_handler=dh)
    q_test = TestFunction("p", dof_handler=dh)

    coeff = phi_k - Constant(1.0)
    expr = q_test * dot(grad(dalpha) * coeff + grad(alpha_k) * dphi, v_k)
    form = expr * dx(metadata={"q": 6})

    equation = Equation(form, None)
    K_py, F_py = assemble_form(equation, dof_handler=dh, bcs=[], quad_order=6, backend="python")
    K_backend, F_backend = assemble_form(equation, dof_handler=dh, bcs=[], quad_order=6, backend=backend)

    A_py = K_py.tocsr().toarray()
    A_backend = K_backend.tocsr().toarray()
    dA = float(np.max(np.abs(A_backend - A_py)))
    dF = float(np.max(np.abs(np.asarray(F_backend, dtype=float) - np.asarray(F_py, dtype=float))))

    assert np.isfinite(dA)
    assert np.isfinite(dF)
    assert dA < 1.0e-9
    assert dF < 1.0e-10
