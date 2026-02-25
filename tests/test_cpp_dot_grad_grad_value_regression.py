import numpy as np


def test_cpp_dot_grad_grad_value_matches_inner(tmp_path, monkeypatch):
    # Ensure a clean kernel cache so the test is not affected by any user-local JIT artifacts.
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_jit_cache"))

    from pycutfem.core.dofhandler import DofHandler
    from pycutfem.core.mesh import Mesh
    from pycutfem.fem.mixedelement import MixedElement
    from pycutfem.ufl.expressions import Function, dot, grad, inner
    from pycutfem.ufl.measures import dx
    from pycutfem.utils.meshgen import structured_quad

    from examples.utils.biofilm.adhesion import assemble_scalar

    poly_order = 2
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=poly_order)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )

    me = MixedElement(mesh, field_specs={"S": 1, "alpha": 1})
    dh = DofHandler(me, method="cg")
    S = Function("S", "S", dof_handler=dh)
    alpha = Function("alpha", "alpha", dof_handler=dh)

    # S=y, alpha=y => grad(S)=(0,1), grad(alpha)=(0,1) so:
    #   dot(grad(S), grad(alpha)) = 1 everywhere, integral = area = 1.
    coords = np.asarray(dh.get_dof_coords("S"), dtype=float)
    S.nodal_values[:] = coords[:, 1]
    alpha.nodal_values[:] = coords[:, 1]

    qdeg = 6
    dx_q = dx(metadata={"q": qdeg})
    I_dot = dot(grad(S), grad(alpha)) * dx_q
    I_inner = inner(grad(S), grad(alpha)) * dx_q

    val_dot = assemble_scalar(dh, I_dot, backend="cpp", quad_order=qdeg)
    val_inner = assemble_scalar(dh, I_inner, backend="cpp", quad_order=qdeg)

    assert np.isfinite(val_dot)
    assert np.isfinite(val_inner)
    assert np.allclose(val_dot, val_inner, rtol=1e-12, atol=1e-12)
    assert np.allclose(val_dot, 1.0, rtol=1e-12, atol=1e-12)

