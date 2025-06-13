import numpy as np, pytest
from pycutfem.utils.meshgen   import structured_quad
from pycutfem.core.mesh       import Mesh
from pycutfem.core.levelset   import CircleLevelSet
from ufl.measures    import dInterface
from ufl.expressions import Constant, Pos, Neg, Jump, FacetNormal,grad, Function, dot, inner
from ufl.forms           import assemble_form
from pycutfem.core.dofhandler import DofHandler

L     = 2.0
R     = 0.7
center= (L/2, L/2)

@pytest.fixture
def mesh():
    nodes, elems, _, corners = structured_quad(L, L, nx=40, ny=40, poly_order=1)
    return Mesh(nodes=nodes, element_connectivity=elems,elements_corner_nodes= corners,element_type= 'quad', poly_order=1)

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

def add_vector_field(mesh:Mesh, phi:CircleLevelSet):
    vals = np.zeros((len(mesh.nodes_list), 2))
    for node in mesh.nodes_list:
        if phi((node.x,node.y))>=0:
            vals[node.id] = [node.y, 3*node.y]
        else:
            vals[node.id] = [2*node.y,4*node.y]
    mesh.node_data_vec = {'v': vals}
    return

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

    # Create the positive-side function
    u_pos_func = Function(field_name='u', name='u_pos',
                          nodal_values=np.zeros(len(mesh.nodes_list)))
    for node in mesh.nodes_list:
        u_pos_func.nodal_values[node.id] = u_pos(node.x, node.y)

    # Create the negative-side function
    u_neg_func = Function(field_name='u', name='u_neg',
                          nodal_values=np.zeros(len(mesh.nodes_list)))
    for node in mesh.nodes_list:
        u_neg_func.nodal_values[node.id] = u_neg(node.x, node.y)


    # --- Define the form using the two functions ---
    n = FacetNormal()
    
    # The form now correctly represents the jump between two distinct fields
    form = dot(grad(u_pos_func) - grad(u_neg_func), n) * dInterface(level_set=phi)

    # --- Assembly and Assertion ---
    eq = form == Constant(0.0)
    hook = {type(form.integrand): {'name': 'gj'}}
    res = assemble_form(eq, DofHandler({'u': mesh}), bcs=[], assembler_hooks=hook)
    
    print(res)
    # This will now pass, as g_pos - g_neg will correctly evaluate to zero.
    assert np.isclose(res['gj'], 0.0, atol=1e-2)

# ------------------------------------------------ gradient jump scalar
def test_jump_grad_scalar(mesh:Mesh):
    phi = make_levelset(); mesh.classify_elements(phi); mesh.classify_edges(phi)
    mesh.build_interface_segments(phi); 
    u_pos = lambda x,y: x**2       # outside  (φ ≥ 0)
    u_neg = lambda x,y: x**2-1     # inside   (φ < 0)
    u_scalar = Function(field_name='u', name='u_scalar',
                        nodal_values=np.zeros(len(mesh.nodes_list)))
    u_scalar = add_scalar_field(u_scalar,mesh, phi, u_pos, u_neg)
    print(u_scalar.nodal_values)

    n   = FacetNormal()                    # unit normal from ctx
    grad_jump = Jump(grad(u_scalar))        # placeholder: visit_Jump will supply 1
    form = dot(grad_jump , n) * dInterface(level_set=phi)
    eq   = form == Constant(0.0)
    hook = {type(form.integrand):{'name':'gj'}}
    res  = assemble_form(eq, DofHandler({'u':mesh}), bcs=[],assembler_hooks=hook)
    print(res)
    assert np.isclose(res['gj'], 0.0, atol=1e-2)

# ------------------------------------------------ vector value jump (norm)
# def test_jump_vector_norm(mesh:Mesh):
#     phi = make_levelset(); mesh.classify_elements(phi); mesh.classify_edges(phi)
#     mesh.build_interface_segments(phi); add_vector_field(mesh, phi)

#     v_pos = Pos(Constant(lambda x,y: np.array([y,3*y])))
#     v_neg = Neg(Constant(lambda x,y: np.array([2*y,4*y])))
#     jump_norm = np.sqrt((v_pos - v_neg)[0]**2 + (v_pos - v_neg)[1]**2)
#     form = jump_norm * dInterface(level_set=phi)
#     eq   = form == Constant(0.0)
#     res  = assemble_form(eq, DofHandler({'v':mesh}), bcs=[],
#                          assembler_hooks={type(form.integrand):{'name':'jv'}})
#     exact = 4*R*np.sqrt(2)
#     assert np.isclose(res['jv'], exact, rtol=1e-2)

if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])