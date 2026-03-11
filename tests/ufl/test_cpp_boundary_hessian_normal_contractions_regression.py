import numpy as np

from pycutfem.core.dofhandler import DofHandler
from pycutfem.core.mesh import Mesh
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import FacetNormal, Function, Hessian, TestFunction as ScalarTestFunction, TrialFunction, VectorFunction, VectorTestFunction, VectorTrialFunction, dot, inner
from pycutfem.ufl.forms import Equation, assemble_form
from pycutfem.ufl.measures import dS
from pycutfem.ufl.spaces import FunctionSpace
from pycutfem.utils.meshgen import structured_quad


def _build_boundary_scalar_problem():
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=4, ny=4, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    mesh.tag_boundary_edges({"all": lambda x, y: True})
    me = MixedElement(mesh, field_specs={"p": 2})
    dh = DofHandler(me, method="cg")

    p = TrialFunction("p", dof_handler=dh)
    q = ScalarTestFunction("p", dof_handler=dh)
    p_k = Function(name="p_k", field_name="p", dof_handler=dh)
    p_k.set_values_from_function(lambda x, y: np.sin(1.7 * x) + 0.2 * np.cos(2.3 * y))

    n = FacetNormal()
    dSb = dS(mesh.edge_bitset("all"), metadata={"q": 4})
    a = inner(dot(n, dot(Hessian(p), n)), dot(n, dot(Hessian(q), n))) * dSb
    l = inner(dot(n, dot(Hessian(p_k), n)), dot(n, dot(Hessian(q), n))) * dSb
    return dh, a, l


def _build_boundary_vector_problem():
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=4, ny=4, poly_order=2)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elems,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=2,
    )
    mesh.tag_boundary_edges({"all": lambda x, y: True})
    me = MixedElement(mesh, field_specs={"ux": 2, "uy": 2})
    dh = DofHandler(me, method="cg")

    space = FunctionSpace(name="u", field_names=["ux", "uy"], dim=1)
    u = VectorTrialFunction(space=space, dof_handler=dh)
    v = VectorTestFunction(space=space, dof_handler=dh)
    u_k = VectorFunction(name="u_k", field_names=["ux", "uy"], dof_handler=dh)
    u_k.set_values_from_function(
        lambda x, y: np.asarray([np.sin(1.3 * x) + 0.1 * y, np.cos(1.1 * y) - 0.2 * x], dtype=float)
    )

    n = FacetNormal()
    dSb = dS(mesh.edge_bitset("all"), metadata={"q": 4})
    a = inner(dot(Hessian(u), n), dot(Hessian(v), n)) * dSb
    l = inner(dot(Hessian(u_k), n), dot(Hessian(v), n)) * dSb
    return dh, a, l


def test_cpp_boundary_scalar_nhn_hessian_parity(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_jit_cache"))
    dh, a, l = _build_boundary_scalar_problem()

    k_py, f_py = assemble_form(Equation(a, l), dof_handler=dh, bcs=[], backend="python")
    k_cpp, f_cpp = assemble_form(Equation(a, l), dof_handler=dh, bcs=[], backend="cpp")

    diff_k = (k_py.tocsr() - k_cpp.tocsr()).tocoo()
    max_k = float(np.max(np.abs(diff_k.data))) if diff_k.nnz else 0.0
    max_f = float(np.max(np.abs(np.asarray(f_py) - np.asarray(f_cpp))))

    assert max_k < 1.0e-10
    assert max_f < 1.0e-10


def test_cpp_boundary_vector_hdotn_hessian_parity(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PYCUTFEM_CACHE_DIR", str(tmp_path / "pycutfem_jit_cache"))
    dh, a, l = _build_boundary_vector_problem()

    k_py, f_py = assemble_form(Equation(a, l), dof_handler=dh, bcs=[], backend="python")
    k_cpp, f_cpp = assemble_form(Equation(a, l), dof_handler=dh, bcs=[], backend="cpp")

    diff_k = (k_py.tocsr() - k_cpp.tocsr()).tocoo()
    max_k = float(np.max(np.abs(diff_k.data))) if diff_k.nnz else 0.0
    max_f = float(np.max(np.abs(np.asarray(f_py) - np.asarray(f_cpp))))

    assert max_k < 1.0e-10
    assert max_f < 1.0e-10
