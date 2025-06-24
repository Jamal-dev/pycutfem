# test_stokes_mixed.py
import numpy as np
import scipy.sparse.linalg as spla
import pytest

# --- Core ---
from pycutfem.core.mesh      import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.utils.meshgen  import structured_quad

# --- UFL-like front-end ---
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions   import (
    TrialFunction, TestFunction, VectorTrialFunction, VectorTestFunction,
    grad, inner, dot, div, Constant
)
from pycutfem.ufl.measures      import dx
from pycutfem.ufl.forms         import BoundaryCondition, assemble_form

# ----------------------------------------------------------------------
#  Utilities common to both cases
# ----------------------------------------------------------------------
def _make_th_dofhandler(nx: int, ny: int) -> tuple[DofHandler, Mesh]:
    """Create geometry mesh (Q2), MixedElement Q2–Q2–Q1 and its DofHandler."""
    nodes, elems, _, corners = structured_quad(1, 1, nx=nx, ny=ny, poly_order=2)
    mesh_q2 = Mesh(nodes, elems, elements_corner_nodes=corners,
                   element_type="quad", poly_order=2)

    # Q2 for velocity components, Q1 (order 1) for pressure
    me = MixedElement(mesh_q2, ('ux', 'uy', 'p'), field_orders={'p': 1})
    dh = DofHandler(me, method='cg')
    return dh, mesh_q2


# ======================================================================
#  1. Lid-driven cavity (Q2–Q1)
# ======================================================================
def test_stokes_lid_driven_cavity():
    dh, mesh = _make_th_dofhandler(4, 4)

    # --- Function spaces & symbols ------------------------------------
    vel_space = FunctionSpace("velocity", ['ux', 'uy'], dim=1)
    pres_space = FunctionSpace("pressure", ['p'], dim=0)

    u = VectorTrialFunction(vel_space)
    v = VectorTestFunction(vel_space)
    p = TrialFunction('p');  q = TestFunction('p')

    # Weak form: ∇u:∇v − p ∇·v + q ∇·u = f·v
    a = (inner(grad(u), grad(v)) - p*div(v) + q*div(u)) * dx()
    f = Constant([0.0, 0.0], dim=1)
    L = dot(f, v) * dx()
    equation = a == L

    # --- Boundary conditions ------------------------------------------
    walls = {
        'bottom': lambda x,y: np.isclose(y, 0),
        'left'  : lambda x,y: np.isclose(x, 0),
        'right' : lambda x,y: np.isclose(x, 1),
        'top'   : lambda x,y: np.isclose(y, 1)
    }
    mesh.tag_boundary_edges(walls)
    mesh.nodes_list[0].tag = 'pressure_pin'      # pin p at (0,0)

    bcs = [
        # No-slip on three walls
        *[BoundaryCondition(c, 'dirichlet', w, 0.0)
          for c in ('ux','uy') for w in ('left','right','bottom')],
        # Moving lid
        BoundaryCondition('ux', 'dirichlet', 'top', 1.0),
        BoundaryCondition('uy', 'dirichlet', 'top', 0.0),
        # Pressure pin
        BoundaryCondition('p', 'dirichlet', 'pressure_pin', 0.0)
    ]

    # --- Assemble & solve ---------------------------------------------
    K, F = assemble_form(equation, dh, bcs=bcs, quad_order=5)
    assert np.linalg.matrix_rank(K.toarray()) == K.shape[0]
    sol = spla.spsolve(K, F)

    # --- Checks --------------------------------------------------------
    ux_top_dof = dh.dof_map['ux'][mesh.nodes_list[-5].id]   # a top-edge node
    assert np.isclose(sol[ux_top_dof], 1.0)

    p_pin = dh.dof_map['p'][mesh.nodes_list[0].id]
    assert np.isclose(sol[p_pin], 0.0)

    print("✓ Lid-driven cavity (Q2–Q1) passed.")


# ======================================================================
# 2. Couette flow (vector form, Q2–Q1)
# ======================================================================
def test_stokes_couette_flow_vector_form():
    dh, mesh = _make_th_dofhandler(nx=4, ny=8)

    vel_space = FunctionSpace("velocity", ['ux', 'uy'], dim=1)
    pres_space = FunctionSpace("pressure", ['p'], dim=0)

    u = VectorTrialFunction(vel_space)
    v = VectorTestFunction(vel_space)
    p = TrialFunction('p');  q = TestFunction('p')

    a = (inner(grad(u), grad(v)) - p*div(v) + q*div(u)) * dx()
    f = Constant([0.0, 0.0], dim=1)
    L = dot(f, v) * dx()
    equation = a == L

    # Couette BCs: stationary bottom, lid speed = 1
    walls = {'bottom': lambda x,y: np.isclose(y,0),
             'top'   : lambda x,y: np.isclose(y,1)}
    mesh.tag_boundary_edges(walls)
    mesh.nodes_list[0].tag = 'pressure_pin'

    bcs = [
        BoundaryCondition('ux','dirichlet','bottom',0.0),
        BoundaryCondition('uy','dirichlet','bottom',0.0),
        BoundaryCondition('ux','dirichlet','top',1.0),
        BoundaryCondition('uy','dirichlet','top',0.0),
        BoundaryCondition('p', 'dirichlet', 'pressure_pin', 0.0)
    ]

    K, F = assemble_form(equation, dh, bcs=bcs, quad_order=5)
    assert np.linalg.matrix_rank(K.toarray()) == K.shape[0]
    sol = spla.spsolve(K, F)

    # --- Analytical profile ux = y, uy = 0 -----------------------------
    tol = 1e-9
    for nd in mesh.nodes_list:
        y = nd.y
        ux = sol[dh.dof_map['ux'][nd.id]]
        uy = sol[dh.dof_map['uy'][nd.id]]
        if not (np.isclose(y,0) or np.isclose(y,1)):
            assert np.isclose(ux, y, atol=tol)
        assert np.isclose(uy, 0.0, atol=tol)

    print("✓ Couette flow profile matches analytical solution.")


# ----------------------------------------------------------------------
if __name__ == '__main__':
    pytest.main([__file__])
