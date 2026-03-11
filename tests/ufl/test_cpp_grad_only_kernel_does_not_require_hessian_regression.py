import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.jit.cpp_backend import compile_backend_cpp
from pycutfem.ufl.expressions import CellDiameter, Constant, FacetNormal, Jump, Neg, Pos, TestFunction as ScalarTestFunction, TrialFunction, dot, grad, inner
from pycutfem.ufl.measures import ds, dx
from pycutfem.utils.meshgen import structured_quad


def test_cpp_grad_only_volume_kernel_does_not_request_hessian(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_jit_cache"))

    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    u = TrialFunction("u", dof_handler=dh)
    v = ScalarTestFunction("u", dof_handler=dh)
    form = inner(grad(u), grad(v)) * dx(metadata={"q": 4})

    runner, _ = compile_backend_cpp(form.integrand, dh, me, on_facet=form.measure.on_facet)

    assert "Hxi0" not in runner.param_order
    assert "Hxi1" not in runner.param_order
    assert "pos_Hxi0" not in runner.param_order
    assert "neg_Hxi0" not in runner.param_order


def test_cpp_grad_only_interior_facet_kernel_does_not_request_hessian(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_jit_cache"))

    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=1)
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
    v = ScalarTestFunction("u", dof_handler=dh)
    n = FacetNormal()
    h = CellDiameter()
    penalty = Constant(10.0)

    avg_grad_u = 0.5 * (Pos(grad(u)) + Neg(grad(u)))
    jump_v = Jump(v)
    form = (-dot(avg_grad_u, n) * jump_v + penalty / h * Jump(u) * jump_v) * ds(metadata={"q": 4})

    runner, _ = compile_backend_cpp(form.integrand, dh, me, on_facet=form.measure.on_facet)

    assert "Hxi0" not in runner.param_order
    assert "Hxi1" not in runner.param_order
    assert "pos_Hxi0" not in runner.param_order
    assert "neg_Hxi0" not in runner.param_order
