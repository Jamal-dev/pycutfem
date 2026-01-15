import numpy as np
import scipy.sparse as sp

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import VectorFunction, VectorTestFunction, VectorTrialFunction, div, dot, grad
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.measures import dx
from pycutfem.utils.meshgen import structured_quad


def _as_dense(mat) -> np.ndarray:
    if sp.issparse(mat):
        return mat.toarray()
    return np.asarray(mat, dtype=float)


def test_cpp_matches_jit_for_dot_gradv_trial_dot_uk():
    """
    Regression: C++ backend mis-compiled the contraction

      dot(dot(grad(v_test), u_trial), u_k)

    by treating an intermediate (component-stacked) matrix list as a gradient stack
    and applying a spatial dot (dot_grad_basis_vector) instead of a component-axis
    contraction (contract_first_first).
    """
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, field_specs={"ux": 1, "uy": 1})
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["ux", "uy"])
    u_trial = VectorTrialFunction(space=V, dof_handler=dh)
    v_test = VectorTestFunction(space=V, dof_handler=dh)

    u_k = VectorFunction(name="u_k", field_names=["ux", "uy"], dof_handler=dh)
    ux_coords = dh.get_dof_coords("ux")
    uy_coords = dh.get_dof_coords("uy")
    u_k.set_component_values(0, np.ones(len(ux_coords), dtype=float))
    u_k.set_component_values(1, 2.0 * np.ones(len(uy_coords), dtype=float))

    form = dot(dot(grad(v_test), u_trial), u_k) * dx(metadata={"q": 4})

    K_jit, _ = assemble_form(Equation(form, None), dof_handler=dh, bcs=[], backend="jit")
    K_cpp, _ = assemble_form(Equation(form, None), dof_handler=dh, bcs=[], backend="cpp")

    A_jit = _as_dense(K_jit)
    A_cpp = _as_dense(K_cpp)

    assert A_cpp.shape == A_jit.shape
    assert np.isfinite(A_cpp).all()
    assert np.allclose(A_cpp, A_jit, rtol=1.0e-10, atol=1.0e-12)


def test_python_matches_jit_for_div_uk_times_dot_uk_v():
    """
    Regression: python backend lacked support for scalar Function × vector test functions
    in products like div(u_k) * dot(u_k, v_test). This scalar×vector-test pattern can
    appear in stabilized formulations and in older "skew+div" convection variants.
    """
    nodes, elems, edges, corners = structured_quad(1.0, 1.0, nx=2, ny=2, poly_order=1)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=1,
    )
    me = MixedElement(mesh, field_specs={"ux": 1, "uy": 1})
    dh = DofHandler(me, method="cg")

    V = FunctionSpace("V", ["ux", "uy"])
    v_test = VectorTestFunction(space=V, dof_handler=dh)
    u_k = VectorFunction(name="u_k", field_names=["ux", "uy"], dof_handler=dh)
    ux_coords = dh.get_dof_coords("ux")
    uy_coords = dh.get_dof_coords("uy")
    u_k.set_component_values(0, np.ones(len(ux_coords), dtype=float))
    u_k.set_component_values(1, 2.0 * np.ones(len(uy_coords), dtype=float))

    rhs = div(u_k) * dot(u_k, v_test) * dx(metadata={"q": 4})

    _, f_py = assemble_form(Equation(None, rhs), dof_handler=dh, bcs=[], backend="python")
    _, f_jit = assemble_form(Equation(None, rhs), dof_handler=dh, bcs=[], backend="jit")

    f_py = np.asarray(f_py, dtype=float).reshape(-1)
    f_jit = np.asarray(f_jit, dtype=float).reshape(-1)

    assert f_py.shape == f_jit.shape
    assert np.isfinite(f_py).all()
    assert np.allclose(f_py, f_jit, rtol=1.0e-10, atol=1.0e-12)
