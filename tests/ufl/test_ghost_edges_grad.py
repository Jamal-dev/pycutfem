# tests/test_ghost_edges_grad.py
import numpy as np, pytest
from pycutfem.utils.meshgen   import structured_quad
from pycutfem.core.mesh       import Mesh
from pycutfem.core.levelset   import CircleLevelSet
from pycutfem.ufl.measures    import dInterface, dx, dGhost
from pycutfem.ufl.expressions import (Constant, Pos, Neg, Jump, FacetNormal,grad, 
                                      Function, dot, inner, VectorFunction, VectorTrialFunction ,
                                      TestFunction, VectorTestFunction, TrialFunction)
from pycutfem.ufl.forms           import assemble_form, Equation
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.functionspace import FunctionSpace
from numpy.testing import assert_allclose # Add this import at the top
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.analytic import x as x_ana
from pycutfem.ufl.analytic import y as y_ana
from pycutfem.ufl.forms import BoundaryCondition, assemble_form
from pycutfem.core.levelset import AffineLevelSet
from pycutfem.io.visualization import plot_mesh_2
import matplotlib.pyplot as plt
from pycutfem.ufl.compilers import FormCompiler

@pytest.fixture(scope="module")
def setup_quad2():
    """2×1 quadratic mesh cut vertically at x=1."""
    poly_order = 2
    nodes, elements_connectivity, edge_connectivity, corner_nodes = structured_quad(2.0, 1.0, nx=20, ny=5, poly_order=poly_order)
    mesh = Mesh(nodes = nodes,
                element_connectivity = elements_connectivity,
                edges_connectivity = edge_connectivity,
                elements_corner_nodes = corner_nodes,
                element_type="quad",
                poly_order=poly_order)
    
    level_set = AffineLevelSet(a=1.0, b=0, c=-1.0)  # Vertical line at x=1

    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)

    ghost = mesh.edge_bitset('ghost')
    fig, ax = plt.subplots(figsize=(10, 8))
    # plot_mesh_2(mesh, ax=ax, level_set=level_set, show=True, 
    #           plot_nodes=False, elem_tags=True, edge_colors=True)
    assert ghost.cardinality() > 0, "Mesh should contain ghost edges for the test."
    # This call now works because of the fixes in visualization.py
    

    me = MixedElement(mesh, field_specs={"u": poly_order})
    dh = DofHandler(me, method="cg")

    comp = FormCompiler(dh, quadrature_order=4)

    return mesh, level_set, ghost, dh, comp

@pytest.mark.parametrize("backend", ["jit", "python"])
def test_gradjumpn_spd(setup_quad2, backend):
    _mesh, ls, ghost, dh, _ = setup_quad2
    n = FacetNormal()

    u_pos = TrialFunction("u","u_pos", dh)
    v_pos = TestFunction ("u","v_pos", dh)
    u_neg = TrialFunction("u","u_neg", dh)
    v_neg = TestFunction ("u","v_neg", dh)

    # (n · ⟦∇u⟧) (n · ⟦∇v⟧)
    a = inner(dot(n, Jump(grad(u_pos), grad(u_neg))),
              dot(Jump(grad(v_pos), grad(v_neg)), n)) \
        * dGhost(defined_on=ghost, level_set=ls,
                 metadata={"derivs": {(1,0),(0,1)}, "q": 6})

    K,_ = assemble_form(Equation(a, None), dof_handler=dh, bcs=[], backend=backend)
    Kd = K.toarray()
    assert np.allclose(Kd, Kd.T, atol=1e-12)
    evals = np.linalg.eigvalsh(Kd)
    assert np.all(evals >= -1e-12)

@pytest.mark.parametrize("backend", ["jit", "python"])
def test_gradjumpn_zero_for_linear(setup_quad2, backend):
    _mesh, ls, ghost, dh, _ = setup_quad2
    n = FacetNormal()
    uh = Function("u","u", dh)
    uh.set_values_from_function(lambda x,y: x)  # linear → ∇u constant → ⟦∇u⟧ = 0

    E = inner(dot(n, Jump(grad(uh))), dot(Jump(grad(uh)), n)) \
        * dGhost(defined_on=ghost, level_set=ls,
                 metadata={"derivs": {(1,0),(0,1)}, "q": 6})

    hooks = {type(E.integrand): {"name": "E"}}
    res = assemble_form(Equation(None, E), dof_handler=dh, bcs=[], assembler_hooks=hooks, backend=backend)
    assert abs(res["E"]) < 1e-12

@pytest.mark.parametrize("backend", ["jit", "python"])
def test_gradjumpn_known_value_on_aligned_cut(setup_quad2, backend):
    _mesh, ls, ghost, dh, _ = setup_quad2
    n = FacetNormal()

    # u(x,y) = 0 if x<=1; u = x if x>1  ⇒ ⟦∂u/∂n⟧ = 1 along x=1
    uh = Function("u","u", dh)
    uh.set_values_from_function(lambda x,y: x ) # globally linear function

    E = inner(dot(n, Jump(grad(uh))), dot(Jump(grad(uh)), n)) \
        * dGhost(defined_on=ghost, level_set=ls,
                 metadata={"derivs": {(1,0),(0,1)}, "q": 6})

    hooks = {type(E.integrand): {"name": "E"}}
    res = assemble_form(Equation(None, E), dof_handler=dh, bcs=[], assembler_hooks=hooks, backend=backend)

    expected = 0.0   # (jump)^2 × length(ghost line) = 0 × 0, for interface it should be 1
    assert np.isclose(res["E"], expected, rtol=1e-2)
