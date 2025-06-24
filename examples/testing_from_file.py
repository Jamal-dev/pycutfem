import numpy as np
import pandas as pd
import os
import glob  # Import the glob module

# pycutfem imports
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    TrialFunction, TestFunction, VectorTrialFunction, VectorTestFunction,
    Function, VectorFunction, Constant, grad, inner, dot, div
)
from pycutfem.ufl.measures import dx
from pycutfem.ufl.forms import assemble_form

# Other imports
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# --- Problem Setup ---

def setup_pycutfem_problem():
    """
    Sets up the pycutfem problem, defining the mesh, function spaces,
    and all necessary UFL-like objects. This version is independent of FEniCSx.
    """
    # Create meshes for Q2 (velocity) and Q1 (pressure) elements
    nodes_q2, elems_q2, _, corners_q2 = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=2)
    mesh_q2 = Mesh(nodes=nodes_q2, element_connectivity=elems_q2, elements_corner_nodes=corners_q2, element_type="quad", poly_order=2)
    
    nodes_q1, elems_q1, _, corners_q1 = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    mesh_q1 = Mesh(nodes=nodes_q1, element_connectivity=elems_q1, elements_corner_nodes=corners_q1, element_type="quad", poly_order=1)
    
    # Map fields to meshes
    fe_map_pc = {'ux': mesh_q2, 'uy': mesh_q2, 'p': mesh_q1}
    
    # Create DoF handler
    dof_handler_pc = DofHandler(fe_map_pc, method='cg')
    
    # Create pycutfem UFL-like objects
    pc = {
        'du': VectorTrialFunction(FunctionSpace("velocity", ['ux', 'uy'])),
        'dp': TrialFunction(FunctionSpace("pressure", ['p'])),
        'v': VectorTestFunction(FunctionSpace("velocity", ['ux', 'uy'])),
        'q': TestFunction(FunctionSpace("pressure", ['p'])),
        'u_k': VectorFunction(name="u_k", field_names=['ux', 'uy'], dof_handler=dof_handler_pc),
        'p_k': Function(name="p_k", field_name='p', dof_handler=dof_handler_pc),
        'u_n': VectorFunction(name="u_n", field_names=['ux', 'uy'], dof_handler=dof_handler_pc),
        'p_n': Function(name="p_n", field_name='p', dof_handler=dof_handler_pc),
        'rho': Constant(1.0),
        'dt': Constant(0.1),
        'theta': Constant(0.5),
        'mu': Constant(1.0e-2),
        'c': Constant([0.5, -0.2], dim=1) # Note: dim=1 for a vector constant in pycutfem
    }
    
    logging.info("pycutfem problem setup complete.")
    return pc, dof_handler_pc

def initialize_pycutfem_functions(pc, dof_handler_pc):
    """
    Initializes the function data with pseudo-random values.
    Uses a fixed seed to ensure reproducibility.
    """
    logging.info("Initializing function data with reproducible random values.")
    np.random.seed(1234)
    
    # Generate random data for all DoFs
    u_k_p_k_data_pc = np.random.rand(dof_handler_pc.total_dofs)
    u_n_p_n_data_pc = np.random.rand(dof_handler_pc.total_dofs)
    
    # Get DoF slices for each field
    dofs_ux_pc = dof_handler_pc.get_field_slice('ux')
    dofs_uy_pc = dof_handler_pc.get_field_slice('uy')
    dofs_p_pc = dof_handler_pc.get_field_slice('p')
    
    # Assign the data to the corresponding function objects
    pc['u_k'].nodal_values[dofs_ux_pc] = u_k_p_k_data_pc[dofs_ux_pc]
    pc['u_k'].nodal_values[dofs_uy_pc] = u_k_p_k_data_pc[dofs_uy_pc]
    pc['p_k'].nodal_values[:] = u_k_p_k_data_pc[dofs_p_pc]
    
    pc['u_n'].nodal_values[dofs_ux_pc] = u_n_p_n_data_pc[dofs_ux_pc]
    pc['u_n'].nodal_values[dofs_uy_pc] = u_n_p_n_data_pc[dofs_uy_pc]
    pc['p_n'].nodal_values[:] = u_n_p_n_data_pc[dofs_p_pc]
    logging.info("Function data initialized.")

# --- Comparison Logic ---

def compare_term_from_file(term_name, J_pc, R_pc, reference_filepath, rtol=1e-8, atol=1e-8):
    """
    Compares a pycutfem-assembled Jacobian or Residual with a reference value
    loaded from a specific file path.
    """
    print("\n" + f"--- Comparing Term: {term_name} ---")

    # --- Compare Residual Vector ---
    if R_pc is not None:
        logging.info(f"Loading reference residual from: {reference_filepath}")
        try:
            df = pd.read_excel(reference_filepath)
            R_reference = df['fx_reordered_residual'].to_numpy()
        except (FileNotFoundError, KeyError) as e:
            print(f"❌ ERROR: Could not load reference residual for '{term_name}'.")
            print(f"   Details: {e}")
            return

        R_pc_flat = R_pc.flatten()
        try:
            np.testing.assert_allclose(R_pc_flat, R_reference, rtol=rtol, atol=atol)
            print(f"✅ Residual vector for '{term_name}' matches the reference file.")
        except AssertionError as e:
            print(f"❌ Residual vector for '{term_name}' does NOT match the reference file!")
            print(e)

    # --- Compare Jacobian Matrix ---
    if J_pc is not None:
        logging.info(f"Loading reference Jacobian from: {reference_filepath}")
        try:
            J_reference_df = pd.read_excel(reference_filepath, sheet_name='fenics')
            J_reference = J_reference_df.to_numpy()
        except FileNotFoundError as e:
            print(f"❌ ERROR: Could not load reference Jacobian for '{term_name}'.")
            print(f"   Details: {e}")
            return

        J_pc_dense = J_pc.toarray()
        try:
            np.testing.assert_allclose(J_pc_dense, J_reference, rtol=rtol, atol=atol)
            print(f"✅ Jacobian matrix for '{term_name}' matches the reference file.")
        except AssertionError as e:
            print(f"❌ Jacobian matrix for '{term_name}' does NOT match the reference file!")
            print(e)


# ==============================================================================
#                      MAIN TEST HARNESS
# ==============================================================================
if __name__ == '__main__':
    pc, dof_handler_pc = setup_pycutfem_problem()
    initialize_pycutfem_functions(pc, dof_handler_pc)

    # --- Define a map of all possible UFL forms pycutfem can assemble ---
    # The keys are "safe names" that match the file naming convention.
    jacobian_pc = (
        pc['rho'] * dot(pc['du'], pc['v']) / pc['dt'] +
        pc['theta'] * pc['rho'] * dot(dot(grad(pc['u_k']), pc['du']), pc['v']) +
        pc['theta'] * pc['rho'] * dot(dot(grad(pc['du']), pc['u_k']), pc['v']) +
        pc['theta'] * pc['mu'] * inner(grad(pc['du']), grad(pc['v'])) -
        pc['dp'] * div(pc['v']) +
        pc['q'] * div(pc['du'])
    ) * dx()
    residual_pc = (
        pc['rho'] * dot(pc['u_k'] - pc['u_n'], pc['v']) / pc['dt'] +
        pc['theta'] * pc['rho'] * dot(dot(grad(pc['u_k']), pc['u_k']), pc['v']) +
        (1.0 - pc['theta']) * pc['rho'] * dot(dot(grad(pc['u_n']), pc['u_n']), pc['v']) +
        pc['theta'] * pc['mu'] * inner(grad(pc['u_k']), grad(pc['v'])) +
        (1.0 - pc['theta']) * pc['mu'] * inner(grad(pc['u_n']), grad(pc['v'])) -
        pc['p_k'] * div(pc['v']) -
        pc['q'] * div(pc['u_k'])
    ) * dx()

    forms_map = {
        'lhs_mass':          {'pc': pc['rho'] * dot(pc['du'], pc['v']) / pc['dt'] * dx(), 'mat': True, 'deg': 4, 'name': "LHS Mass"},
        'lhs_diffusion':     {'pc': pc['theta'] * pc['mu'] * inner(grad(pc['du']), grad(pc['v'])) * dx(), 'mat': True, 'deg': 4, 'name': "LHS Diffusion"},
        'lhs_advection_1':   {'pc': dot(dot(grad(pc['du']), pc['u_k']), pc['v']) * dx(), 'mat': True, 'deg': 5, 'name': "LHS Advection 1"},
        'lhs_advection_2':   {'pc': pc['theta'] * pc['rho'] * dot(dot(grad(pc['u_k']), pc['du']), pc['v']) * dx(), 'mat': True, 'deg': 5, 'name': "LHS Advection 2"},
        'lhs_pressure':      {'pc': -pc['dp'] * div(pc['v']) * dx(), 'mat': True, 'deg': 3, 'name': "LHS Pressure"},
        'lhs_continuity':    {'pc': pc['q'] * div(pc['du']) * dx(), 'mat': True, 'deg': 3, 'name': "LHS Continuity"},
        'rhs_time_derivative': {'pc': (pc['rho'] * dot(pc['u_k'] - pc['u_n'], pc['v']) / pc['dt']) * dx(), 'mat': False, 'deg': 4, 'name': "RHS Time Derivative"},
        'rhs_advection':     {'pc': pc['theta'] * pc['rho'] * dot(dot(grad(pc['u_k']), pc['u_k']), pc['v']) * dx(), 'mat': False, 'deg': 5, 'name': "RHS Advection"},
        'lhs_scalar_advection': {'pc': dot(grad(pc['dp']), pc['u_k']) * pc['q'] * dx(), 'mat': True, 'deg': 3, 'name': "LHS Scalar Advection"},
        'lhs_vector_advection_constant': {'pc': dot(dot(grad(pc['du']), pc['c']), pc['v']) * dx(), 'mat': True, 'deg': 5, 'name': "LHS Vector Advection Constant"},
        'navier_stokes_lhs': {'pc': jacobian_pc, 'mat': True, 'deg': 5, 'name': "Navier Stokes LHS"},
        'navier_stokes_rhs': {'pc': residual_pc, 'mat': False, 'deg': 5, 'name': "Navier Stokes RHS"},
        'rhs_advection_2':   {'pc': (1.0 - pc['theta']) * pc['rho'] * dot(dot(grad(pc['u_n']), pc['u_n']), pc['v']) * dx(), 'mat': False, 'deg': 5, 'name': "RHS Advection 2"},
        'rhs_mass_matrix':   {'pc': pc['rho'] * dot(pc['u_k'] - pc['u_n'], pc['v']) / pc['dt'] * dx(), 'mat': False, 'deg': 4, 'name': "RHS mass matrix"},
        'rhs_pressure_term': {'pc': pc['p_k'] * div(pc['v']) * dx(), 'mat': False, 'deg': 5, 'name': "RHS pressure term"},
        'rhs_continuity':    {'pc': pc['q'] * div(pc['u_k']) * dx(), 'mat': False, 'deg': 6, 'name': "RHS Continuity"}
    }

    # --- Discover and Run Tests ---
    reference_dir = "garbage"
    reference_files = glob.glob(os.path.join(reference_dir, '*.xlsx'))

    if not reference_files:
        raise FileNotFoundError(f"No reference .xlsx files found in '{reference_dir}'. "
                                "Run the original comparison script to generate them.")

    logging.info(f"Found {len(reference_files)} reference files to test against.")

    for filepath in reference_files:
        basename = os.path.basename(filepath)
        J_pc, R_pc = None, None
        
        # Parse filename to get term name and type (Jacobian/Residual)
        if basename.endswith('_jacobian.xlsx'):
            safe_term_name = basename[:-len('_jacobian.xlsx')]
            is_mat = True
        elif basename.endswith('_residual.xlsx'):
            safe_term_name = basename[:-len('_residual.xlsx')]
            is_mat = False
        else:
            logging.warning(f"Skipping unrecognized file format: {basename}")
            continue
            
        # Find the corresponding form definition
        if safe_term_name not in forms_map:
            logging.warning(f"Skipping test for '{safe_term_name}'. No matching UFL form defined in forms_map.")
            continue
        
        form_info = forms_map[safe_term_name]

        # Assemble the pycutfem form
        if form_info['mat']:
            J_pc, _ = assemble_form(form_info['pc'] == Constant(0.0) * pc['v'][0] * dx(), dof_handler_pc, quad_degree=form_info['deg'])
        else:
            _, R_pc = assemble_form(Constant(0.0) * pc['v'][0] * dx() == form_info['pc'], dof_handler_pc, quad_degree=form_info['deg'])
        
        # Compare the result with the saved reference file
        compare_term_from_file(form_info['name'], J_pc, R_pc, reference_filepath=filepath)

    print("\n" + "="*70)
    print("Data-driven regression testing complete.")
    print("="*70)
