import pytest
import numpy as np

# --- Core imports ---
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.core.levelset import CircleLevelSet
from pycutfem.utils.bitset import BitSet

# --- UFL-like imports ---
from ufl.expressions import Constant, jump, Pos, Neg
from ufl.measures import ds
from ufl.forms import assemble_form

def test_perimeter_calculation():
    """
    Tests _assemble_face_integral by computing the perimeter of a circle.
    This validates the core loop over cut edges and geometric terms.
    """
    # 1. Setup a mesh and a level set
    mesh_size = 2.0
    nx = 20 # Use a reasonably fine mesh
    nodes, elems, _, corners = structured_quad(mesh_size, mesh_size, nx=nx, ny=nx, poly_order=1)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=1)

    radius = 0.7
    level_set = CircleLevelSet(center=(0.0, 0.0), radius=radius)
    
    # 2. Use the robust mesh classification methods
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)

    # Create a BitSet of interface edges for the ds() measure
    is_interface_edge = [edge.tag == 'interface' for edge in mesh.edges_list]
    interface_edges = BitSet(is_interface_edge)
    assert interface_edges.cardinality() > 0, "No interface edges were found by mesh.classify_edges."

    # 3. Define the form for the perimeter
    form = Constant(1.0) * ds(defined_on=interface_edges, level_set=level_set)
    dummy_equation = form == Constant(0.0)
    
    # 4. Assemble, using a hook to capture the scalar result
    dof_handler = DofHandler({'phi': mesh}) # Dummy handler
    hooks = { Constant: {'name': 'perimeter'} }
    
    results = assemble_form(dummy_equation, dof_handler=dof_handler, bcs=[])
    
    # 5. Verify the result
    computed_perimeter = results['perimeter']
    exact_perimeter = 2 * np.pi * radius
    
    print(f"\nExact Perimeter: {exact_perimeter:.6f}")
    print(f"Computed Perimeter: {computed_perimeter:.6f}")
    
    assert np.isclose(computed_perimeter, exact_perimeter, rtol=1e-2), \
        "Computed perimeter is not close to the exact value."

    print("Perimeter calculation test passed.")

def test_jump_of_constant():
    """
    Tests the jump of a discontinuous constant field.
    The form is Integral(jump(u) * ds), where u is conceptually { C1 if phi>0, C2 if phi<0 }.
    This validates the Pos(u) and Neg(u) visitors.
    """
    # 1. Setup mesh and level set, same as perimeter test
    mesh_size = 2.0; nx = 20
    nodes, elems, _, corners = structured_quad(mesh_size, mesh_size, nx=nx, ny=nx, poly_order=1)
    mesh = Mesh(nodes=nodes, element_connectivity=elems, elements_corner_nodes=corners, element_type="quad", poly_order=1)
    radius = 0.7
    level_set = CircleLevelSet(center=(0.0, 0.0), radius=radius)
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    is_interface_edge = [edge.tag == 'interface' for edge in mesh.edges_list]
    interface_edges = BitSet(is_interface_edge)

    # 2. Define the form using Pos() and Neg() to create the jump
    # Let u = 5.0 on the '+' side (outside circle) and u = 2.0 on the '-' side (inside)
    # The jump is u('+') - u('-') = 5.0 - 2.0 = 3.0
    C_pos = Constant(5.0)
    C_neg = Constant(2.0)
    
    # We define the jump manually for this test to be explicit
    integrand = Pos(C_pos) - Neg(C_neg)
    form = integrand * ds(defined_on=interface_edges, level_set=level_set)
    dummy_equation = form == Constant(0.0)

    # 3. Assemble with a hook
    dof_handler = DofHandler({'phi': mesh})
    # The hook needs to be on the top-level expression type, which is Sub
    from ufl.expressions import Sub as SubExpr
    hooks = { SubExpr: {'name': 'jump_integral'} }
    results = assemble_form(dummy_equation, dof_handler=dof_handler, bcs=[])
    
    # 4. Verify
    computed_integral = results['jump_integral']
    exact_integral = (5.0 - 2.0) * (2 * np.pi * radius) # jump_value * perimeter
    
    print(f"\nExact Integral of Jump: {exact_integral:.6f}")
    print(f"Computed Integral of Jump: {computed_integral:.6f}")
    
    assert np.isclose(computed_integral, exact_integral, rtol=1e-2)
    print("Constant jump test passed.")


if __name__ == "__main__":
    test_perimeter_calculation()
    test_jump_of_constant()
