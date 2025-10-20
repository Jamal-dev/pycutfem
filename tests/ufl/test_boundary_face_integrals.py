# tests/test_boundary_face_integrals.py
import numpy as np, pytest
from numpy.testing import assert_allclose

from pycutfem.utils.meshgen    import structured_quad
from pycutfem.core.mesh        import Mesh
from pycutfem.ufl.measures     import dS, dx
from pycutfem.ufl.expressions  import Constant, Function, VectorFunction, TrialFunction, TestFunction, VectorTestFunction
from pycutfem.ufl.expressions  import inner, FacetNormal, dot
from pycutfem.ufl.analytic     import Analytic
from pycutfem.ufl.analytic     import y as y_ana
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.forms        import assemble_form, BoundaryCondition,Equation
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.core.dofhandler  import DofHandler

L = 2.0

@pytest.fixture(scope="module")
def mesh_and_dh():
    nodes, elems, _, corners = structured_quad(L, L, nx=20, ny=20, poly_order=1)
    bc_tags = {
        "right_wall":  lambda x,y: np.isclose(x, L),
        "left_wall":   lambda x,y: np.isclose(x, 0),
        "top_wall":    lambda x,y: np.isclose(y, L),
        "bottom_wall": lambda x,y: np.isclose(y, 0),
    }
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners,
                element_type="quad", poly_order=1)
    mesh.tag_boundary_edges(bc_tags)

    me  = MixedElement(mesh, field_specs={"u": 1})
    dh  = DofHandler(me, method="cg")
    return mesh, dh
@pytest.fixture(scope="module")
def mesh_vec_and_dh():
    nodes, elems, _, corners = structured_quad(L, L, nx=20, ny=20, poly_order=1)
    bc_tags = {
        "right_wall":  lambda x,y: np.isclose(x, L),
        "left_wall":   lambda x,y: np.isclose(x, 0),
        "top_wall":    lambda x,y: np.isclose(y, L),
        "bottom_wall": lambda x,y: np.isclose(y, 0),
    }
    mesh = Mesh(nodes, elems, elements_corner_nodes=corners,
                element_type="quad", poly_order=1)
    mesh.tag_boundary_edges(bc_tags)

    me  = MixedElement(mesh, field_specs={"vx": 1,"vy": 1})
    dh  = DofHandler(me, method="cg")
    return mesh, dh

# ----------------------------------------------------------------------





@pytest.mark.parametrize("backend", ["python","jit"])
def test_y_moment_right_edge(mesh_and_dh,backend):
    mesh, dh = mesh_and_dh
    form = Analytic(y_ana) * dS(mesh.edge_bitset("right_wall"))
    res  = assemble_form(Equation(form, None),
                         dof_handler=dh, assembler_hooks={type(form.integrand):{"name":"m"}}, backend=backend)
    assert_allclose(res["m"], 0.5*L**2, rtol=1e-13)

# tests/ufl/test_boundary_face_integrals.py
def test_len_right_edge_function(mesh_and_dh):
    mesh, dh   = mesh_and_dh
    u          = Function(name='u', field_name='u', dof_handler=dh); 
    u.set_values_from_function(lambda x,y: 2.0*y)
    form       = u * dS(mesh.edge_bitset("right_wall"))
    exact      = L**2        # ∫_0^L   2y dy  = L²
    res        = assemble_form(Equation(form, None),
                               dof_handler=dh,
                               assembler_hooks={type(form.integrand): {'name':'len_f'}},
                               backend="jit")
    assert_allclose(res['len_f'], exact, rtol=1e-13)

@pytest.mark.parametrize("backend", ["python","jit"])
def test_len_vectorfun(mesh_vec_and_dh,backend):
    print("Testing length of vector function on right edge with backend:", backend)
    mesh, dh   = mesh_vec_and_dh
    v = VectorFunction(name="v", field_names=["vx","vy"], dof_handler=dh)
    # fs = FunctionSpace(name="velocity", field_names=["vx", "vy"], dim=1)
    # v_test = VectorTestFunction(space=fs, dof_handler=dh)
    v.set_values_from_function(lambda x,y: np.array([y, 3*y]))
    n = FacetNormal()
    dot_c = dot(v, n)
    form = dot_c * dS(mesh.edge_bitset("right_wall"),metadata={"q":3})
    # rhs = dot(Constant(np.asarray([0.0, 0.0]), dim=1) , v_test) * dx
    res  = assemble_form(Equation(form, None), dh,
            assembler_hooks={dot_c:{"name":"flux"}},
            backend=backend)
    assert_allclose(res["flux"], 0.5*L**2, rtol=1e-13)

@pytest.mark.parametrize("backend", ["python","jit"])
def test_bilinear(mesh_and_dh, backend):
    mesh, dh   = mesh_and_dh
    u = TrialFunction("u", dof_handler=dh)
    v = TestFunction ("u", dof_handler=dh)
    form = inner(u, v) * dS(mesh.edge_bitset("right_wall"))
    _   = assemble_form(
                        Equation(form, None),
                        dh, backend=backend)  # just needs to run
    
@pytest.mark.parametrize("backend", ["python","jit"])
def test_len_boundary_right_edge(mesh_and_dh,backend):
    mesh, dh = mesh_and_dh
    c_1 = Constant(1.0)
    form = c_1 * dS(mesh.edge_bitset("right_wall"),metadata={"q":3})
    v = TestFunction ("u", dof_handler=dh)
    res  = assemble_form(Equation(None, form),
                         dof_handler=dh, assembler_hooks={c_1:{"name":"len"}}, backend=backend)
    assert_allclose(res["len"], L, rtol=1e-13)