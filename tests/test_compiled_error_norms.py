import numpy as np
import pytest
from numpy.testing import assert_allclose

from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.expressions import Function, VectorFunction
from pycutfem.core.levelset import AffineLevelSet


def _have_cpp_backend() -> bool:
    try:
        from pycutfem.jit.cpp_backend import compile_backend_cpp  # noqa: F401
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def dh_and_level_set():
    # Ω = [-0.5,0.5]^2, split by x = 0.1
    L = 1.0
    nodes, elems, _, corners = structured_quad(L, L, nx=8, ny=8, poly_order=1, offset=[-0.5, -0.5])
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners, element_type="quad", poly_order=1)

    me = MixedElement(mesh, field_specs={"u": 1, "ux": 1, "uy": 1})
    dh = DofHandler(me, method="cg")

    level_set = AffineLevelSet(1.0, 0.0, -0.1)  # φ(x,y)=x-0.1
    mesh.classify_elements(level_set)
    return dh, level_set


@pytest.mark.parametrize("backend", ["jit"] + (["cpp"] if _have_cpp_backend() else []))
def test_l2_error_on_side_compiled_matches_python(dh_and_level_set, backend):
    dh, level_set = dh_and_level_set

    uh = Function(name="uh", field_name="u", dof_handler=dh)

    exact = {"u": lambda x, y: x + 2.0 * y}
    q = 6

    err_py = dh.l2_error_on_side(
        function=uh,
        exact=exact,
        level_set=level_set,
        side="-",
        fields=["u"],
        quad_order=q,
        relative=False,
    )
    err_cmp = dh.l2_error_on_side_compiled(
        function=uh,
        exact=exact,
        level_set=level_set,
        side="-",
        fields=["u"],
        quad_order=q,
        relative=False,
        backend=backend,
    )
    assert_allclose(err_cmp, err_py, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("backend", ["jit"] + (["cpp"] if _have_cpp_backend() else []))
def test_h1_scalar_error_compiled_matches_python_and_exact(dh_and_level_set, backend):
    dh, level_set = dh_and_level_set
    uh = Function(name="uh", field_name="u", dof_handler=dh)

    def exact_grad(x, y):
        return np.array([1.0, 2.0], dtype=float)

    err_py = dh.h1_error_scalar_on_side(
        uh,
        exact_grad,
        level_set,
        side="-",
        field="u",
        relative=False,
        quad_increase=0,
    )
    err_cmp = dh.h1_error_scalar_on_side_compiled(
        uh,
        exact_grad,
        level_set,
        side="-",
        field="u",
        relative=False,
        quad_increase=0,
        backend=backend,
    )

    # Ω⁻ has area 0.6, ||grad||² = 1²+2² = 5  =>  sqrt(5*0.6) = sqrt(3)
    err_exact = float(np.sqrt(3.0))
    assert_allclose(err_cmp, err_py, rtol=1e-11, atol=1e-12)
    assert_allclose(err_cmp, err_exact, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("backend", ["jit"] + (["cpp"] if _have_cpp_backend() else []))
def test_h1_scalar_error_compiled_zero_for_exact_linear(dh_and_level_set, backend):
    dh, level_set = dh_and_level_set
    uh = Function(name="uh", field_name="u", dof_handler=dh)

    # Exact Q1 field: u = x + 2y -> grad(u) = [1, 2]
    uh.set_values_from_function(lambda x, y: x + 2.0 * y)

    def exact_grad(x, y):
        return np.array([1.0, 2.0], dtype=float)

    err_py = dh.h1_error_scalar_on_side(
        uh,
        exact_grad,
        level_set,
        side="-",
        field="u",
        relative=False,
        quad_increase=0,
    )
    err_cmp = dh.h1_error_scalar_on_side_compiled(
        uh,
        exact_grad,
        level_set,
        side="-",
        field="u",
        relative=False,
        quad_increase=0,
        backend=backend,
    )

    assert err_py < 1e-10
    assert_allclose(err_cmp, err_py, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("backend", ["jit"] + (["cpp"] if _have_cpp_backend() else []))
def test_h1_vector_error_compiled_matches_python_and_exact(dh_and_level_set, backend):
    dh, level_set = dh_and_level_set
    Uh = VectorFunction(name="U", field_names=["ux", "uy"], dof_handler=dh)

    def exact_grad_vec(x, y):
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    q = 2
    err_py = dh.h1_error_vector_on_side(
        Uh,
        exact_grad_vec,
        level_set,
        side="-",
        fields=["ux", "uy"],
        relative=False,
        quad_increase=0,
        quad_order=q,
    )
    err_cmp = dh.h1_error_vector_on_side_compiled(
        Uh,
        exact_grad_vec,
        level_set,
        side="-",
        fields=["ux", "uy"],
        relative=False,
        quad_increase=0,
        quad_order=q,
        backend=backend,
    )

    # Ω⁻ has area 0.6, ||grad||² = 1 + 1 = 2  =>  sqrt(2*0.6)
    err_exact = float(np.sqrt(1.2))
    assert_allclose(err_cmp, err_py, rtol=1e-11, atol=1e-12)
    assert_allclose(err_cmp, err_exact, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("backend", ["jit"] + (["cpp"] if _have_cpp_backend() else []))
def test_h1_vector_error_compiled_zero_for_exact_linear(dh_and_level_set, backend):
    dh, level_set = dh_and_level_set
    Uh = VectorFunction(name="U", field_names=["ux", "uy"], dof_handler=dh)

    # Exact Q1 vector field:
    #   U = (x + y, 2x - y) -> grad(U) = [[1, 1], [2, -1]]
    Uh.set_values_from_function(lambda x, y: np.array([x + y, 2.0 * x - y], dtype=float))

    def exact_grad_vec(x, y):
        return np.array([[1.0, 1.0], [2.0, -1.0]], dtype=float)

    q = 6
    err_py = dh.h1_error_vector_on_side(
        Uh,
        exact_grad_vec,
        level_set,
        side="-",
        fields=["ux", "uy"],
        relative=False,
        quad_increase=0,
        quad_order=q,
    )
    err_cmp = dh.h1_error_vector_on_side_compiled(
        Uh,
        exact_grad_vec,
        level_set,
        side="-",
        fields=["ux", "uy"],
        relative=False,
        quad_increase=0,
        quad_order=q,
        backend=backend,
    )

    assert err_py < 1e-10
    assert_allclose(err_cmp, err_py, rtol=1e-11, atol=1e-12)
