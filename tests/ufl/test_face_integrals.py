import numpy as np, pytest
from pycutfem.utils.meshgen   import structured_quad
from pycutfem.core.mesh       import Mesh
from pycutfem.core.levelset   import CircleLevelSet
from pycutfem.ufl.measures    import dInterface
from pycutfem.ufl.expressions import Constant, Pos, Neg, Jump, FacetNormal,grad, Function, dot, inner, VectorFunction
from pycutfem.ufl.forms           import assemble_form
from pycutfem.core.dofhandler import DofHandler
from pycutfem.ufl.functionspace import FunctionSpace
from numpy.testing import assert_allclose # Add this import at the top


L     = 2.0
R     = 0.7
center= (L/2, L/2)

@pytest.fixture
def mesh(poly_order=1):
    nodes, elems, _, corners = structured_quad(L, L, nx=40, ny=40, poly_order=poly_order)
    return Mesh(nodes=nodes, element_connectivity=elems,elements_corner_nodes= corners,element_type= 'quad', poly_order=poly_order)

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
        func.nodal_values[node.id] = value
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
def test_jump_scalar(mesh:Mesh):
    phi = make_levelset()
    mesh.classify_elements(phi); mesh.classify_edges(phi); 
    mesh.build_interface_segments(phi)
    # add_scalar_field(mesh, phi)
    u_pos = Constant(lambda x,y: x)       # outside  (φ ≥ 0)
    u_neg = Constant(lambda x,y: 2*x)     # inside   (φ < 0)
    jump = Jump(u_pos , u_neg)           # jump expression

    # u_out = x     ;  u_in = 2x
    # u_pos = Pos(Constant(lambda x,y: x))   # evaluate via lambda inside visitor
    # u_neg = Neg(Constant(lambda x,y: 2*x))
    form = jump * dInterface(level_set=phi)
    eq    = form == Constant(0.0)

    res = assemble_form(eq, DofHandler({'u':mesh}), bcs=[],
                        assembler_hooks={type(form.integrand):{'name':'jmp'}})
    J = res['jmp']
    print(f"Jump scalar value: {J}")
    exact = - 2 * np.pi * phi.radius * center[0]  # integral of jump over interface
    assert np.isclose(J, exact, atol=1e-2)

def test_jump_grad_scalar_manual(mesh:Mesh):
    phi = make_levelset(); mesh.classify_elements(phi); mesh.classify_edges(phi)
    mesh.build_interface_segments(phi); 
    dof_handler = DofHandler({'u':mesh})  # Create a DofHandler for the mesh
    u_pos = lambda x,y: x**2       # outside  (φ ≥ 0)
    u_neg = lambda x,y: x**2-1     # inside   (φ < 0)
    grad_u_pos_x = lambda x,y: 2*x       # grad outside
    grad_u_neg_x = lambda x,y: 2*x       # grad inside
    grad_u_pos_y = lambda x,y: 0         # grad outside
    grad_u_neg_y = lambda x,y: 0         # grad inside
    jump_grad_u = Jump(Constant( [grad_u_pos_x, grad_u_pos_y],dim=1),
                        Constant([grad_u_neg_x, grad_u_neg_y],dim=1))  # jump in vector
    n   = FacetNormal()                    # unit normal from ctx
    form = dot(jump_grad_u,n) * dInterface(level_set=phi) 
    
    eq   = form == Constant(0.0)
    hook = {type(form.integrand):{'name':'gj'}}
    res  = assemble_form(eq, DofHandler({'u':mesh}), bcs=[],assembler_hooks=hook)
    print(res)
    assert np.isclose(res['gj'], 0.0, atol=1e-2)


def test_jump_grad_scalar_two_fields(mesh: Mesh):
    phi = make_levelset()
    # ... setup mesh ...

    # --- Create TWO separate, smooth Function objects ---
    u_pos = lambda x, y: x**2
    u_neg = lambda x, y: x**2 - 1

    dof_handler = DofHandler({'u': mesh})  # Create a DofHandler for the mesh
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
    
    # The form now correctly represents the jump between two distinct fields
    form = dot(grad(u_pos_func) - grad(u_neg_func), n) * dInterface(level_set=phi)

    # --- Assembly and Assertion ---
    eq = form == Constant(0.0)
    hook = {type(form.integrand): {'name': 'gj'}}
    res = assemble_form(eq, dof_handler=dof_handler, bcs=[], assembler_hooks=hook)
    
    print(res)
    # This will now pass, as g_pos - g_neg will correctly evaluate to zero.
    assert np.isclose(res['gj'], 0.0, atol=1e-2)

# ------------------------------------------------ gradient jump scalar
def test_jump_grad_scalar(mesh:Mesh):
    phi = make_levelset(); mesh.classify_elements(phi); mesh.classify_edges(phi)
    mesh.build_interface_segments(phi); 
    u_pos = lambda x,y: x**2       # outside  (φ ≥ 0)
    u_neg = lambda x,y: x**2-1     # inside   (φ < 0)
    dof_handler = DofHandler({'u':mesh})  # Create a DofHandler for the mesh
    u_scalar = Function(field_name='u', name='u_scalar',
                        dof_handler=dof_handler)
    u_scalar = add_scalar_field(u_scalar,mesh, phi, u_pos, u_neg)

    n   = FacetNormal()                    # unit normal from ctx
    grad_jump = Jump(grad(u_scalar))        # placeholder: visit_Jump will supply 1
    form = dot(grad_jump , n) * dInterface(level_set=phi)
    eq   = form == Constant(0.0)
    hook = {type(form.integrand):{'name':'gj'}}
    res  = assemble_form(eq, dof_handler=dof_handler, bcs=[],assembler_hooks=hook)
    print(res)
    assert np.isclose(res['gj'], 0.0, atol=1e-2)

# ------------------------------------------------ vector value jump (norm)
def test_jump_vector_norm(mesh:Mesh):
    fe_map = {'vx': mesh, 'vy':mesh}  # Define a vector field
    dof_handler = DofHandler(fe_map, method='cg')
    velocity_space_pos = VectorFunction("velocity_pos", ['vx', 'vy'], dof_handler=dof_handler)
    velocity_space_neg = VectorFunction("velocity_neg", ['vx', 'vy'], dof_handler=dof_handler)
    phi = make_levelset(); mesh.classify_elements(phi); mesh.classify_edges(phi)
    mesh.build_interface_segments(phi); 
    v_pos = lambda x,y: np.array([y,3*y])
    v_neg = lambda x,y: np.array([2*y,4*y])
    velocity_space_pos.set_values_from_function(v_pos)
    velocity_space_neg.set_values_from_function(v_neg)
    n = FacetNormal()  # unit normal from ctx
    jump_v = Jump(velocity_space_pos,velocity_space_neg)  # jump in vector

    form = dot(jump_v,n) * dInterface(level_set=phi)
    eq   = form == Constant(0.0)
    res  = assemble_form(eq, dof_handler, bcs=[],
                         assembler_hooks={type(form.integrand):{'name':'jv'}})
    exact = -R**2 * np.pi
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
    fe_map = {'vx': mesh, 'vy':mesh}  # Define a vector field
    dof_handler = DofHandler(fe_map, method='cg')
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
    eq   = form == Constant(0.0)
    res  = assemble_form(eq, dof_handler=dof_handler, bcs=[],
                         assembler_hooks={type(form.integrand):{'name':'jv'}})
    exact = reference_solution_vector()
    print(f"Exact vector jump: {exact}")
    print(f"Computed vector jump: {res['jv']}")
    assert_allclose(res['jv'], exact.flatten(), rtol=1e-2)

if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])