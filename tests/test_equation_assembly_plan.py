import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.jit import _FusedIntegralGroup, _plan_equation_assembly
from pycutfem.ufl.compilers import FormCompiler
from pycutfem.ufl.expressions import Constant, TestFunction, TrialFunction
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad


def _build_scalar_equation():
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

    a = du * u_test * dx + Constant(2.0) * du * u_test * dx
    L = Constant(3.0) * u_test * dx + Constant(-0.5) * u_test * dx
    return dh, Equation(a, L)


def test_equation_plan_schedules_rhs_before_lhs_for_cpp(monkeypatch) -> None:
    dh, eq = _build_scalar_equation()
    compiler = FormCompiler(dh, backend="cpp")

    monkeypatch.setenv("PYCUTFEM_CPP_FUSE_INTEGRALS", "1")
    plan = _plan_equation_assembly(
        eq,
        compiler=compiler,
        p_geo=1,
        backend="cpp",
        need_matrix=True,
        need_vector=True,
        vector_first=True,
    )

    assert [stage.target for stage in plan.stages] == ["vector", "matrix"]
    assert len(plan.vector_units) == 1
    assert len(plan.matrix_units) == 1
    assert isinstance(plan.vector_units[0], _FusedIntegralGroup)
    assert isinstance(plan.matrix_units[0], _FusedIntegralGroup)
    assert tuple(plan.vector_units[0].integrals) == tuple(eq.L.integrals)
    assert tuple(plan.matrix_units[0].integrals) == tuple(eq.a.integrals)


def test_form_compiler_residual_only_matches_full_rhs(monkeypatch) -> None:
    dh, eq = _build_scalar_equation()
    monkeypatch.setenv("PYCUTFEM_FUSE_INTEGRALS", "0")

    _, residual_full = FormCompiler(dh, backend="python").assemble(eq, bcs=[], need_matrix=True)
    matrix_none, residual_only = FormCompiler(dh, backend="python").assemble(eq, bcs=[], need_matrix=False)

    assert matrix_none is None
    np.testing.assert_allclose(residual_only, residual_full, rtol=1.0e-12, atol=1.0e-12)


def test_form_compiler_executes_rhs_stage_before_lhs(monkeypatch) -> None:
    dh, eq = _build_scalar_equation()
    monkeypatch.setenv("PYCUTFEM_FUSE_INTEGRALS", "0")

    calls: list[tuple[str, int]] = []

    def _record_volume(self, integral, target):
        del target
        calls.append(("rhs" if bool(self.ctx.get("rhs", False)) else "lhs", int(id(integral))))

    monkeypatch.setattr(FormCompiler, "_assemble_volume", _record_volume)
    FormCompiler(dh, backend="python").assemble(eq, bcs=[], need_matrix=True)

    expected = [("rhs", int(id(intg))) for intg in eq.L.integrals]
    expected.extend(("lhs", int(id(intg))) for intg in eq.a.integrals)
    assert calls == expected


def test_volume_local_contributions_skip_matrix_when_not_needed(monkeypatch) -> None:
    dh, eq = _build_scalar_equation()
    monkeypatch.setenv("PYCUTFEM_FUSE_INTEGRALS", "0")

    batch_full = FormCompiler(dh, backend="python").assemble_volume_local_contributions(eq, need_matrix=True)
    batch_res = FormCompiler(dh, backend="python").assemble_volume_local_contributions(eq, need_matrix=False)

    assert batch_res.K_elem is None
    np.testing.assert_allclose(batch_res.F_elem, batch_full.F_elem, rtol=1.0e-12, atol=1.0e-12)
