# tests/test_boundary_face_integrals.py
import numpy as np, pytest
from numpy.testing import assert_allclose

from pycutfem.utils.meshgen    import structured_quad
from pycutfem.core.mesh        import Mesh
from pycutfem.ufl.measures     import dS, dx
from pycutfem.ufl.expressions  import Constant
from pycutfem.ufl.analytic     import Analytic
from pycutfem.ufl.analytic     import y as y_ana
from pycutfem.ufl.forms        import assemble_form, BoundaryCondition
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

# ----------------------------------------------------------------------
def test_length_right_edge(mesh_and_dh):
    mesh, dh = mesh_and_dh
    form     = Constant(1.0) * dS(defined_on=mesh.edge_bitset("right_wall"))
    eq       = form == Constant(0.0) * dx
    res      = assemble_form(eq, dh,
                assembler_hooks={type(form.integrand): {"name": "len"}}, backend="python")
    assert_allclose(res["len"], L, rtol=1e-12)

def test_moment_y_on_right_edge(mesh_and_dh):
    mesh, dh = mesh_and_dh
    form = Analytic(y_ana) * dS(defined_on=mesh.edge_bitset("right_wall"))
    eq   = form == Constant(0.0) * dx
    res  = assemble_form(eq, dh,
            assembler_hooks={type(form.integrand): {"name": "moment"}}, backend="python")
    exact = 0.5 * L**2           # ∫₀ᴸ y dy
    assert_allclose(res["moment"], exact, rtol=1e-12)
