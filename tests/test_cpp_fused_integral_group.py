import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.jit import _form_integrals, compile_multi
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import Constant, TestFunction, TrialFunction
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad


def _build_scalar_problem():
    nodes, elems, _, corners = structured_quad(
        1.0,
        1.0,
        nx=1,
        ny=1,
        poly_order=1,
        offset=(0.0, 0.0),
    )
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=1)
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    du = TrialFunction("u", dof_handler=dh)
    u_test = TestFunction("u", dof_handler=dh)

    c_mass_a = Constant(2.0)
    c_mass_b = Constant(-0.75)
    form = du * u_test * dx + c_mass_a * du * u_test * dx + c_mass_b * du * u_test * dx
    return dh, form


def _build_zero_linear_problem():
    nodes, elems, _, corners = structured_quad(
        1.0,
        1.0,
        nx=1,
        ny=1,
        poly_order=1,
        offset=(0.0, 0.0),
    )
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=1)
    me = MixedElement(mesh, field_specs={"u": 1})
    dh = DofHandler(me, method="cg")

    du = TrialFunction("u", dof_handler=dh)
    u_test = TestFunction("u", dof_handler=dh)

    bilinear = du * u_test * dx + Constant(1.5) * du * u_test * dx
    zero_linear = Constant(0.0) * u_test * dx
    return dh, bilinear, zero_linear, bilinear + zero_linear


def _build_disjoint_field_problem():
    nodes, elems, _, corners = structured_quad(
        1.0,
        1.0,
        nx=1,
        ny=1,
        poly_order=1,
        offset=(0.0, 0.0),
    )
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=1)
    me = MixedElement(mesh, field_specs={"u": 1, "v": 1})
    dh = DofHandler(me, method="cg")

    du = TrialFunction("u", dof_handler=dh)
    u_test = TestFunction("u", dof_handler=dh)
    dv = TrialFunction("v", dof_handler=dh)
    v_test = TestFunction("v", dof_handler=dh)

    form = du * u_test * dx + Constant(2.0) * du * u_test * dx + Constant(3.0) * dv * v_test * dx
    return dh, form


def _assemble_matrix(dh, form, *, backend: str):
    matrix, _ = assemble_form(
        Equation(form, None),
        dof_handler=dh,
        bcs=[],
        backend=backend,
    )
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=float)
    return np.asarray(matrix, dtype=float)


def _assemble_local_volume_batch(dh, form, *, backend: str):
    compiler = FormCompiler(dh, backend=backend)
    return compiler.assemble_volume_local_contributions(form)


def test_cpp_shared_loop_fusion_matches_unfused_and_python(monkeypatch) -> None:
    dh, form = _build_scalar_problem()

    monkeypatch.setenv("PYCUTFEM_CPP_FAST_COMPILE", "1")
    monkeypatch.setenv("PYCUTFEM_CPP_FUSE_INTEGRALS", "0")
    kernels_plain = compile_multi(form, dof_handler=dh, mixed_element=dh.mixed_element, backend="cpp")

    monkeypatch.setenv("PYCUTFEM_CPP_FUSE_INTEGRALS", "1")
    kernels_fused = compile_multi(form, dof_handler=dh, mixed_element=dh.mixed_element, backend="cpp")

    assert len(form.integrals) == 3
    assert len(kernels_plain) == 3
    assert len(kernels_fused) == 1
    assert getattr(kernels_fused[0], "fused_integral_ids", []) == [id(intg) for intg in form.integrals]

    matrix_py = _assemble_matrix(dh, form, backend="python")

    monkeypatch.setenv("PYCUTFEM_CPP_FUSE_INTEGRALS", "0")
    matrix_plain = _assemble_matrix(dh, form, backend="cpp")

    monkeypatch.setenv("PYCUTFEM_CPP_FUSE_INTEGRALS", "1")
    matrix_fused = _assemble_matrix(dh, form, backend="cpp")

    np.testing.assert_allclose(matrix_plain, matrix_py, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(matrix_fused, matrix_py, rtol=1.0e-12, atol=1.0e-12)


def test_cpp_shared_loop_local_batches_match_unfused_and_python(monkeypatch) -> None:
    import pycutfem.jit.cpp_backend as cpp_backend

    dh, form = _build_scalar_problem()

    monkeypatch.setenv("PYCUTFEM_CPP_FAST_COMPILE", "1")

    batch_py = _assemble_local_volume_batch(dh, form, backend="python")

    monkeypatch.setenv("PYCUTFEM_CPP_FUSE_INTEGRALS", "0")
    batch_plain = _assemble_local_volume_batch(dh, form, backend="cpp")

    group_calls: list[int] = []
    orig_group = cpp_backend.compile_backend_cpp_group

    def _wrapped_group(exprs, *args, **kwargs):
        group_calls.append(len(exprs))
        return orig_group(exprs, *args, **kwargs)

    monkeypatch.setattr(cpp_backend, "compile_backend_cpp_group", _wrapped_group)
    monkeypatch.setenv("PYCUTFEM_CPP_FUSE_INTEGRALS", "1")
    batch_fused = _assemble_local_volume_batch(dh, form, backend="cpp")

    assert group_calls == [3]
    np.testing.assert_array_equal(batch_plain.gdofs_map, batch_py.gdofs_map)
    np.testing.assert_array_equal(batch_fused.gdofs_map, batch_py.gdofs_map)
    np.testing.assert_allclose(batch_plain.K_elem, batch_py.K_elem, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(batch_fused.K_elem, batch_py.K_elem, rtol=1.0e-12, atol=1.0e-12)


def test_exact_zero_linear_form_builds_no_cpp_kernel(monkeypatch) -> None:
    dh, _, zero_linear, _ = _build_zero_linear_problem()

    monkeypatch.setenv("PYCUTFEM_CPP_FAST_COMPILE", "1")
    monkeypatch.setenv("PYCUTFEM_CPP_FUSE_INTEGRALS", "1")

    assert _form_integrals(zero_linear) == []
    kernels = compile_multi(zero_linear, dof_handler=dh, mixed_element=dh.mixed_element, backend="cpp")
    assert kernels == []

    rhs_py = assemble_form(
        Equation(None, zero_linear),
        dof_handler=dh,
        bcs=[],
        backend="python",
    )[1]
    rhs_cpp = assemble_form(
        Equation(None, zero_linear),
        dof_handler=dh,
        bcs=[],
        backend="cpp",
    )[1]
    np.testing.assert_allclose(np.asarray(rhs_cpp, dtype=float), np.asarray(rhs_py, dtype=float), rtol=1.0e-12, atol=1.0e-12)


def test_zero_linear_integrals_do_not_contaminate_cpp_fusion(monkeypatch) -> None:
    dh, bilinear, _, contaminated = _build_zero_linear_problem()

    monkeypatch.setenv("PYCUTFEM_CPP_FAST_COMPILE", "1")
    monkeypatch.setenv("PYCUTFEM_CPP_FUSE_INTEGRALS", "1")

    kernels = compile_multi(contaminated, dof_handler=dh, mixed_element=dh.mixed_element, backend="cpp")
    assert len(_form_integrals(contaminated)) == 2
    assert len(kernels) == 1
    assert getattr(kernels[0], "fused_integral_ids", []) == [id(intg) for intg in _form_integrals(contaminated)]

    matrix_bilinear = _assemble_matrix(dh, bilinear, backend="python")
    matrix_cpp = _assemble_matrix(dh, contaminated, backend="cpp")
    np.testing.assert_allclose(matrix_cpp, matrix_bilinear, rtol=1.0e-12, atol=1.0e-12)


def test_cpp_shared_loop_keeps_disjoint_field_groups_separate(monkeypatch) -> None:
    dh, form = _build_disjoint_field_problem()

    monkeypatch.setenv("PYCUTFEM_CPP_FAST_COMPILE", "1")
    monkeypatch.setenv("PYCUTFEM_CPP_FUSE_INTEGRALS", "1")

    kernels = compile_multi(form, dof_handler=dh, mixed_element=dh.mixed_element, backend="cpp")

    assert len(form.integrals) == 3
    assert len(kernels) == 2
    assert getattr(kernels[0], "fused_integral_ids", []) == [id(form.integrals[0]), id(form.integrals[1])]
    assert getattr(kernels[1], "fused_integral_ids", []) in ([], None)

    matrix_py = _assemble_matrix(dh, form, backend="python")
    matrix_cpp = _assemble_matrix(dh, form, backend="cpp")
    np.testing.assert_allclose(matrix_cpp, matrix_py, rtol=1.0e-12, atol=1.0e-12)
