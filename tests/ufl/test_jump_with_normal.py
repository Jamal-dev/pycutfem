import os
import numpy as np
import pytest

from pycutfem.core.levelset import AffineLevelSet
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Constant, Dot, FacetNormal, Function, Neg, Pos, TrialFunction, grad, inner, jump
from pycutfem.ufl.expressions import TestFunction as UFLTestFunction
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dCutSkeleton
from examples.utils.fsi.fully_eulerian import make_domain_sets
from pycutfem.utils.meshgen import structured_quad


def _have_numba() -> bool:
    try:
        import numba  # noqa: F401
    except Exception:
        return False
    return True


def _have_pybind11() -> bool:
    try:
        import pybind11  # noqa: F401
    except Exception:
        return False
    return True


def _selected_backends(*, default: str = "python") -> list[str]:
    spec = (os.environ.get("PYCUTFEM_TEST_BACKENDS") or os.environ.get("BACKEND") or default).strip()
    if not spec:
        return [default]
    if spec.lower() in {"all", "*"}:
        return ["python", "jit", "cpp"]
    backends = [b.strip() for b in spec.split(",") if b.strip()]
    valid = {"python", "jit", "cpp"}
    unknown = [b for b in backends if b not in valid]
    if unknown:
        raise ValueError(f"Unknown backend(s) {unknown}; valid={sorted(valid)}")
    return backends


def _skip_if_backend_unavailable(backend: str) -> None:
    if backend == "jit" and not _have_numba():
        pytest.skip("numba not available")
    if backend == "cpp" and not _have_pybind11():
        pytest.skip("pybind11 not available")


@pytest.mark.parametrize("backend", _selected_backends())
def test_jump_with_normal_matches_expansion(backend):
    """
    Regression test:
      - `jump(v, n)` must implement the UFL semantics v(+)·n(+) + v(-)·n(-)
      - Evaluating `dot(grad(Function), n)` must *not* create VecOpInfo with float `.data`

    This is exercised on a cut-skeleton integral, which hits the CutFEM facet code paths.
    """
    _skip_if_backend_unavailable(str(backend))
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=1, offset=(0.0, 0.0))
    mesh = Mesh(nodes, elems, edges, corners, element_type="quad", poly_order=1)
    ls = AffineLevelSet(1.0, 0.0, -0.5)  # phi = x-0.5
    mesh.classify_elements(ls)
    mesh.classify_edges(ls)
    make_domain_sets(mesh, use_aligned_interface=False)

    me = MixedElement(mesh, {"u": 1})
    dh = DofHandler(me, method="cg")

    u_k = Function(name="u_k", field_name="u", dof_handler=dh)
    u_k.set_values_from_function(lambda x, y: np.array([x + 2.0 * y], dtype=float))

    du = TrialFunction(field_name="u", name="du", dof_handler=dh)
    v = UFLTestFunction(field_name="u", name="v", dof_handler=dh)
    n = FacetNormal()
    derivs = {(1, 0), (0, 1)}
    dS = dCutSkeleton(level_set=ls, metadata={"q": 4, "side": "+", "derivs": derivs})

    j_fun = jump(grad(u_k), n)
    j_test = jump(grad(v), n)
    r1 = inner(j_fun, j_test) * dS

    j_fun_exp = Dot(Pos(grad(u_k)), Pos(n)) + Dot(Neg(grad(u_k)), Neg(n))
    j_test_exp = Dot(Pos(grad(v)), Pos(n)) + Dot(Neg(grad(v)), Neg(n))
    r2 = inner(j_fun_exp, j_test_exp) * dS

    # assemble residual-only to exercise function-path evaluation
    eq1 = Equation(Constant(0.0) * du * v * dS, r1)
    eq2 = Equation(Constant(0.0) * du * v * dS, r2)
    _, F1 = assemble_form(eq1, dof_handler=dh, bcs=[], quad_order=4, backend=backend)
    _, F2 = assemble_form(eq2, dof_handler=dh, bcs=[], quad_order=4, backend=backend)

    assert np.allclose(F1, F2, rtol=1.0e-11, atol=1.0e-12)
