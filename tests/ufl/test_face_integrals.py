from matplotlib.pylab import f
import numpy as np, pytest
from pycutfem.utils.meshgen   import structured_quad
from pycutfem.core.mesh       import Mesh
from pycutfem.core.levelset   import CircleLevelSet
from pycutfem.ufl.measures    import dInterface, dx, dGhost
from pycutfem.ufl.expressions import (Constant, Pos, Neg, Jump, FacetNormal,grad, 
                                      Function, dot, inner, VectorFunction, VectorTrialFunction ,
                                      TestFunction, VectorTestFunction, TrialFunction)
from pycutfem.ufl.forms           import assemble_form
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

import logging



L     = 2.0
R     = 0.7
center= (L/2, L/2)

@pytest.fixture(scope="module")
def cavity_setup(poly_order=1):
    """
    A single, heavy-lifting fixture that computes all necessary objects
    and returns them in a tuple. This is efficient as the computation
    is only done once.
    """
    print("\n--- Running cavity_setup fixture ---")
    nodes, elems, _, corners = structured_quad(L, L, nx=40, ny=40, poly_order=poly_order)
    bc_tags = {
        'bottom_wall': lambda x,y: np.isclose(y,0),
        'left_wall':   lambda x,y: np.isclose(x,0),
        'right_wall':  lambda x,y: np.isclose(x,L),
        'top_lid':     lambda x,y: np.isclose(y,L)
    }
    bcs = [
        BoundaryCondition('u', 'dirichlet', 'bottom_wall', lambda x,y: 0.0),
        BoundaryCondition('u', 'dirichlet', 'left_wall',   lambda x,y: 0.0),
        BoundaryCondition('u', 'dirichlet', 'right_wall',  lambda x,y: 0.0),
        BoundaryCondition('u', 'dirichlet', 'top_lid',     lambda x,y: 0.0),
    ]
    mesh_obj = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type='quad', poly_order=poly_order)
    mesh_obj.tag_boundary_edges(bc_tags)
    me = MixedElement(mesh_obj, field_specs={"u": 1})
    dof_handler_obj = DofHandler(me, method='cg')
    
    return mesh_obj, dof_handler_obj, bcs

@pytest.fixture(scope="module")
def mesh(cavity_setup):
    """Gets the mesh object from the main setup fixture."""
    return cavity_setup[0]

@pytest.fixture(scope="module")
def dof_handler(cavity_setup):
    """Gets the DofHandler object from the main setup fixture."""
    return cavity_setup[1]

@pytest.fixture(scope="module")
def bcs(cavity_setup):
    """Gets the boundary conditions list from the main setup fixture."""
    return cavity_setup[2]

def make_levelset():
    return CircleLevelSet(center, radius=R)

def add_scalar_field(func: Function, mesh: Mesh, phi, u_pos=None, u_neg=None):
    """
    Populates the nodal values of a Function object based on the sign of a level set.
    """
    # Assign to the 'nodal_values' array within the Function object
    for node in mesh.nodes_list:
        is_positive = phi((node.x, node.y)) >= 0
        value = u_pos(node.x, node.y) if is_positive else u_neg(node.x, node.y)
        func.set_nodal_values(node.id, value)
    return func

def add_vector_field(vecfun:VectorFunction,mesh:Mesh, phi:CircleLevelSet, v_pos=None, v_neg=None):
    vals = np.zeros((len(mesh.nodes_list), 2))
    for node in mesh.nodes_list:
        if phi((node.x,node.y))>=0:
            vals[node.id] = v_pos(node.x,node.y)
        else:
            vals[node.id] = v_neg(node.x,node.y)
    vecfun.nodal_values = vals
    return vecfun

# ------------------------------------------------ value jump scalar
def test_jump_scalar(mesh:Mesh, dof_handler:DofHandler, bcs:list[BoundaryCondition]):
    phi = make_levelset()
    mesh.classify_elements(phi); mesh.classify_edges(phi); 
    mesh.build_interface_segments(phi)
    # add_scalar_field(mesh, phi)
    u_pos = Analytic(x_ana)       # outside  (φ ≥ 0)
    u_neg = Analytic( 2*x_ana)     # inside   (φ < 0)
    jump = Jump(u_pos , u_neg)           # jump expression

    # u_out = x     ;  u_in = 2x
    # u_pos = Pos(Constant(lambda x,y: x))   # evaluate via lambda inside visitor
    # u_neg = Neg(Constant(lambda x,y: 2*x))
    form = jump * dInterface(level_set=phi,metadata={"q":3})  # dInterface is a measure for the interface
    eq    = form == Constant(0.0) * dx


    res = assemble_form(eq, dof_handler=dof_handler, bcs=[],
                        assembler_hooks={type(form.integrand):{'name':'jmp'}})
    print(f"res: {res}")
    J = res['jmp']
    print(f"Jump scalar value: {J}")
    exact = - 2 * np.pi * phi.radius * center[0]  # integral of jump over interface
    assert np.isclose(J, exact, atol=1e-2)

def test_jump_grad_scalar_manual(mesh:Mesh, dof_handler:DofHandler, bcs:list[BoundaryCondition]):
    phi = make_levelset(); mesh.classify_elements(phi); mesh.classify_edges(phi)
    mesh.build_interface_segments(phi); 
    v = TestFunction('u', dof_handler=dof_handler)  # Test function for the weak form
    u_pos = lambda x,y: x**2       # outside  (φ ≥ 0)
    u_neg = lambda x,y: x**2-1     # inside   (φ < 0)
    grad_u_pos_x = lambda x,y: 2*x       # grad outside
    grad_u_neg_x = lambda x,y: 2*x       # grad inside
    grad_u_pos_y = lambda x,y: 0         # grad outside
    grad_u_neg_y = lambda x,y: 0         # grad inside
    u_func_pos = Function(field_name='u', name='u_pos',
                          dof_handler=dof_handler)
    u_func_pos.set_values_from_function(u_pos)
    u_func_neg = Function(field_name='u', name='u_neg',
                          dof_handler=dof_handler)
    u_func_neg.set_values_from_function(u_neg)
    jump_grad_u = Jump(grad(u_func_pos), grad(u_func_neg))  # jump in gradient
    n   = FacetNormal()                    # unit normal from ctx
    # interface contribution goes on the RHS --------------------
    rhs_form = dot(jump_grad_u, n) * v * dInterface(level_set=phi)

    # a positive-definite mass matrix on the volume --------------
    w = TrialFunction('u', dof_handler=dof_handler)
    lhs_form = w * v * dx

    eq  = lhs_form == rhs_form
    K, F = assemble_form(eq, dof_handler=dof_handler, bcs=bcs)

    u_sol = np.linalg.solve(K.toarray(), F)
    assert np.allclose(u_sol, 0.0, atol=1e-12)




def test_jump_grad_scalar_two_fields(mesh: Mesh, dof_handler: DofHandler, bcs: list[BoundaryCondition]):
    phi = make_levelset()
    # ... setup mesh ...

    # --- Create TWO separate, smooth Function objects ---
    u_pos = lambda x, y: x**2
    u_neg = lambda x, y: x**2 - 1

    # Create the positive-side function
    u_pos_func = Function(field_name='u', name='u_pos',
                          dof_handler=dof_handler)
    u_pos_func.set_values_from_function(u_pos)


    # Create the negative-side function
    u_neg_func = Function(field_name='u', name='u_neg',
                          dof_handler=dof_handler)
    u_neg_func.set_values_from_function(u_neg)


    # --- Define the form using the two functions ---
    n = FacetNormal()
    w = TrialFunction('u', dof_handler=dof_handler)  # Trial function for the weak form
    v = TestFunction('u', dof_handler=dof_handler)  # Test function for the weak form
    # The form now correctly represents the jump between two distinct fields
    rhs = dot(grad(u_pos_func) - grad(u_neg_func), n) * v * dInterface(level_set=phi)

    # --- Assembly and Assertion ---
    eq = w *v * dx == rhs
    # hook = {type(rhs.integrand): {'name': 'gj'}}
    LHS,RHS = assemble_form(eq, dof_handler=dof_handler, bcs=[], assembler_hooks=None)
    res = np.linalg.solve(LHS.toarray(), RHS)
    # This will now pass, as g_pos - g_neg will correctly evaluate to zero.
    assert np.allclose(res, 0.0, atol=1e-10)



# ------------------------------------------------ vector value jump (norm)
def test_jump_vector_norm(mesh:Mesh, dof_handler:DofHandler, bcs:list[BoundaryCondition]):
    
    me = MixedElement(mesh, field_specs={"vx":1, "vy":1})
    dof_handler = DofHandler(me, method='cg')  # Create a DofHandler for the mesh
    velocity_space_pos = VectorFunction(name="velocity_pos", field_names= ['vx', 'vy'], dof_handler=dof_handler)
    velocity_space_neg = VectorFunction(name="velocity_neg", field_names= ['vx', 'vy'], dof_handler=dof_handler)
    phi = make_levelset(); mesh.classify_elements(phi); mesh.classify_edges(phi)
    mesh.build_interface_segments(phi); 
    v_pos = lambda x,y: np.array([y,3*y])
    v_neg = lambda x,y: np.array([2*y,4*y])
    velocity_space_pos.set_values_from_function(v_pos)
    velocity_space_neg.set_values_from_function(v_neg)
    n = FacetNormal()  # unit normal from ctx
    jump_v = Jump(velocity_space_pos,velocity_space_neg)  # jump in vector
    form_lhs = dot(jump_v, n) * dInterface(level_set=phi)
    
    # CORRECTED: RHS must be a valid integral form
    form_rhs = Constant(0.0) * dx
    eq = form_lhs ==  form_rhs

    res = assemble_form(eq, dof_handler, bcs=[],
                        assembler_hooks={type(form_lhs.integrand): {'name': 'jv'}})
    
    exact = -np.pi * R**2
    assert np.isclose(res['jv'], exact, rtol=1e-2)

def reference_solution_vector(L=L, R=R, center=center):
    import sympy as sp
    x, y, theta, r, cx, cy = sp.symbols('x y theta r cx cy')
    v1 = sp.Matrix([2 * x * y, 3 * x**2 * y - y**2])
    v2 = sp.Matrix([3 * y**2 * x + x**3, 10 * x + y])
    grad_v1 = sp.Matrix([[sp.diff(v1[0], x), sp.diff(v1[0], y)],
                         [sp.diff(v1[1], x), sp.diff(v1[1], y)]])
    grad_v2 = sp.Matrix([[sp.diff(v2[0], x), sp.diff(v2[0], y)],
                         [sp.diff(v2[1], x), sp.diff(v2[1], y)]])
    n = sp.Matrix([[sp.cos(theta)], [sp.sin(theta)]])
    dot_product_v1 = grad_v1 * n
    dot_product_v2 = grad_v2 * n
    x_expr = cx + r * sp.cos(theta)
    y_expr = cy + r * sp.sin(theta)
    dot_product_v1_subs = dot_product_v1.subs([(x, x_expr), (y, y_expr)])
    dot_product_v2_subs = dot_product_v2.subs([(x, x_expr), (y, y_expr)])
    jump_expr = dot_product_v1_subs - dot_product_v2_subs
    integral = sp.integrate(r * jump_expr, (theta, 0, 2 * sp.pi))
    integral_subs = integral.subs([(r, R), (cx, center[0]), (cy, center[1])])
    result = np.array(integral_subs).astype(np.float64)
    return result.flatten()

def test_jump_grad_vector(mesh:Mesh):

    me = MixedElement(mesh, field_specs={"vx":1, "vy":1})
    dof_handler = DofHandler(me, method='cg')  # Create a DofHandler for the mesh
    velocity_space_pos = VectorFunction("velocity_pos", ['vx', 'vy'], dof_handler=dof_handler)
    velocity_space_neg = VectorFunction("velocity_neg", ['vx', 'vy'], dof_handler=dof_handler)
    phi = make_levelset(); mesh.classify_elements(phi); mesh.classify_edges(phi)
    mesh.build_interface_segments(phi); 
    v_pos = lambda x,y: np.array([2 * x * y, 3 * x**2 * y-y**2])
    v_neg = lambda x,y: np.array([3 * y**2 * x + x**3, 10 * x +y])
    velocity_space_pos.set_values_from_function(v_pos)
    velocity_space_neg.set_values_from_function(v_neg)
    n = FacetNormal()  # unit normal from ctx
    jump_grad_v = Jump(grad(velocity_space_pos),grad(velocity_space_neg))  # jump in vector
    grad_v_pos_n = dot(grad(velocity_space_pos),n)  
    grad_v_neg_n = dot(grad(velocity_space_neg),n)

    # form = dot(jump_grad_v,n) * dInterface(level_set=phi)
    form = Jump(grad_v_pos_n,grad_v_neg_n) * dInterface(level_set=phi)
    eq   = form == Constant(0.0) * dx
    res  = assemble_form(eq, dof_handler=dof_handler, bcs=[],
                         assembler_hooks={type(form.integrand):{'name':'jv'}})
    exact = reference_solution_vector()
    print(f"Exact vector jump: {exact}")
    print(f"Computed vector jump: {res['jv']}")
    assert_allclose(res['jv'], exact.flatten(), rtol=1e-2)

def assemble_scalar(form, dof_handler):
    """Assemble a scalar functional from a form."""
    # This is a placeholder for the actual assembly logic
    # In practice, this would involve creating an assembler and calling its methods
    hook = {type(form.integrand): {'name': 'scalar'}}
    res = assemble_form(Constant(0.0)*dx==form, dof_handler=dof_handler, bcs=[], backend="jit", assembler_hooks=hook)
    return res["scalar"]


def test_interface_normal_sign_vertical_nonaligned():
    # Mesh: [0,2]x[0,1]
    nodes, elems, _, corners = structured_quad(2, 1, nx=40, ny=20, poly_order=1)
    mesh = Mesh(nodes=nodes, element_connectivity=elems,
                elements_corner_nodes=corners, element_type='quad', poly_order=1)

    # Level set: φ = x - (1 + ε), with ε small so the line cuts element interiors
    eps = 0.007
    ls  = AffineLevelSet(1.0, 0.0, -(1.0 + eps))
    mesh.classify_elements(ls)
    mesh.classify_edges(ls)
    mesh.build_interface_segments(ls)

    me  = MixedElement(mesh, field_specs={"vx": 1, "vy": 1})
    dh  = DofHandler(me, method='cg')
    n   = FacetNormal()

    I = n[0] * dInterface(defined_on=mesh.element_bitset('cut'), level_set=ls)
    val = assemble_scalar(I, dof_handler=dh)

    # The interface is still a vertical segment of height 1, so the integral should be +1
    assert np.isclose(val, 1.0, atol=1e-3)

def test_interface_normal_matches_gradphi_unit():
    nodes, elems, _, corners = structured_quad(2, 1, nx=39, ny=20, poly_order=1)  # nx odd => no face alignment
    mesh = Mesh(nodes=nodes, element_connectivity=elems,
                elements_corner_nodes=corners, element_type='quad', poly_order=1)

    # φ = x - 1 (now x=1 is *not* on a grid line with nx=39)
    ls  = AffineLevelSet(1.0, 0.0, -1.0)
    mesh.classify_elements(ls); mesh.classify_edges(ls); mesh.build_interface_segments(ls)

    me  = MixedElement(mesh, field_specs={"vx":1, "vy":1})
    dh  = DofHandler(me, method='cg')
    n   = FacetNormal()

    # Since ∇φ = (1,0), the unit gradient is just (1,0)
    Ix = n[0] * dInterface(defined_on=mesh.element_bitset('cut'), level_set=ls)
    len_int = dot(n,n) * dInterface(defined_on=mesh.element_bitset('cut'), level_set=ls)
    length = assemble_scalar(len_int, dh)

    val = assemble_scalar(Ix, dh)
    # If your JIT normal is +gradphi/|gradphi|, val == length
    # If it's the outward-fluid normal, val == -length
    print(f"Computed integral value: {val}, expected length: {length}")
    assert np.isclose(abs(val), length, rtol=1e-3)
    # Optional: assert on the *sign* you expect:
    assert val > 0.0     # for the current neg→pos convention

def total_ghost_length(mesh):
    L = 0.0
    for gid in mesh.edge_bitset('ghost').to_indices():
        e = mesh.edge(gid)
        (i, j) = e.nodes
        x0, y0 = mesh.nodes_x_y_pos[i]
        x1, y1 = mesh.nodes_x_y_pos[j]
        L += float(np.hypot(x1-x0, y1-y0))
    return L

def test_ghost_normal_sign_with_gradphi():
    nodes, elems, _, corners = structured_quad(2, 1, nx=39, ny=20, poly_order=1)
    mesh = Mesh(nodes=nodes, element_connectivity=elems,
                elements_corner_nodes=corners, element_type='quad', poly_order=1)
    ls  = AffineLevelSet(1.0, 0.0, -1.0)
    mesh.classify_elements(ls); mesh.classify_edges(ls); mesh.build_interface_segments(ls)
    print(f"Ghost edges: {mesh.edge_bitset('ghost').cardinality()}")
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_mesh_2(mesh, ax=ax, level_set=ls, show=True, 
              plot_nodes=False, elem_tags=True, edge_colors=True)

    me  = MixedElement(mesh, field_specs={"vx":1, "vy":1})
    dh  = DofHandler(me, method='cg')

    n   = FacetNormal()
    dG  = dGhost(defined_on=mesh.edge_bitset('ghost'), level_set=ls)

    val = assemble_scalar( n[0] * dG, dh)        # = ∫_G n_G·(1,0) dG
    Lup = total_ghost_length(mesh)               # upper bound using mesh geometry

    assert val > 0.0
    assert val <= Lup * (1.0 + 1e-3)





if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])