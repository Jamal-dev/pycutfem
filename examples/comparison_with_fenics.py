import numpy as np
import pandas as pd
import os

# FEniCSx imports
from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl
import basix
import basix.ufl
from basix.ufl import mixed_element
from petsc4py import PETSc # Import PETSc for enums like InsertMode

# pycutfem imports
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    TrialFunction, TestFunction, VectorTrialFunction, VectorTestFunction,
    Function, VectorFunction, Constant, grad, inner, dot, div
)
from pycutfem.ufl.measures import dx, dInterface
from pycutfem.ufl.forms import assemble_form
from pycutfem.fem.reference import get_reference
from pycutfem.fem.mixedelement import MixedElement

# Imports for mapping and matrix conversion
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
import logging
logging.basicConfig(
    level=logging.INFO,  # show debug messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Helper functions for coordinates
def get_pycutfem_dof_coords(dof_handler: DofHandler, field: str) -> np.ndarray:
    if field not in dof_handler.field_names:
        raise ValueError(f"Field '{field}' not found in DofHandler")
    mesh = dof_handler.fe_map[field]
    dof_map = dof_handler.dof_map[field]
    dof_coord_pairs = sorted([(g_dof, tuple(mesh.nodes_x_y_pos[n_id])) for n_id, g_dof in dof_map.items()], key=lambda p: p[0])
    return np.array([p[1] for p in dof_coord_pairs])

def get_all_pycutfem_dof_coords(dof_handler: DofHandler) -> np.ndarray:
    all_coords = np.zeros((dof_handler.total_dofs, 2))
    for field in ['ux', 'uy', 'p']:
        field_dofs = dof_handler.get_field_slice(field)
        if hasattr(dof_handler, 'get_dof_coords'):
             field_coords = get_pycutfem_dof_coords(dof_handler, field)
             all_coords[field_dofs] = field_coords
    return all_coords

def get_all_fenicsx_dof_coords(W_fenicsx):
    """Builds the full, ordered list of DoF coordinates for the mixed space."""
    num_total_dofs = W_fenicsx.dofmap.index_map.size_global
    all_coords = np.zeros((num_total_dofs, 2))
    
    W0, V_map = W_fenicsx.sub(0).collapse()
    W1, P_map_fx = W_fenicsx.sub(1).collapse()
    W00, V0_map = W0.sub(0).collapse()
    W01, V1_map = W0.sub(1).collapse()

    coords_ux = W00.tabulate_dof_coordinates()[:, :2]
    coords_uy = W01.tabulate_dof_coordinates()[:, :2]
    coords_p = W1.tabulate_dof_coordinates()[:, :2]

    dofs_ux = np.array(V_map)[np.array(V0_map)]
    dofs_uy = np.array(V_map)[np.array(V1_map)]
    dofs_p = np.array(P_map_fx)

    all_coords[dofs_ux] = coords_ux
    all_coords[dofs_uy] = coords_uy
    all_coords[dofs_p] = coords_p
    
    return all_coords

def one_to_one_map_coords(coords1, coords2):
    C = np.linalg.norm(coords2[:, np.newaxis, :] - coords1[np.newaxis, :, :], axis=2)
    rows, cols = linear_sum_assignment(C)
    return rows[np.argsort(cols)]

def create_true_dof_map(dof_handler_pc, W_fenicsx):
    print("="*70)
    print("Discovering true DoF map by matching DoF coordinates...")
    print("="*70)
    W0, V_map = W_fenicsx.sub(0).collapse()
    W1, P_map_fx = W_fenicsx.sub(1).collapse()
    W00, V0_map = W0.sub(0).collapse()
    W01, V1_map = W0.sub(1).collapse()
    fx_coords = {
        'ux': W00.tabulate_dof_coordinates()[:, :2], 'uy': W01.tabulate_dof_coordinates()[:, :2],
        'p':  W1.tabulate_dof_coordinates()[:, :2]
    }
    V_map_np = np.array(V_map)
    fx_dofs = {
        'ux': V_map_np[np.array(V0_map)], 'uy': V_map_np[np.array(V1_map)],
        'p':  np.array(P_map_fx)
    }
    pc_coords = {f: get_pycutfem_dof_coords(dof_handler_pc, f) for f in ['ux', 'uy', 'p']}
    pc_dofs = {f: dof_handler_pc.get_field_slice(f) for f in ['ux', 'uy', 'p']}
    P = np.zeros(dof_handler_pc.total_dofs, dtype=int)
    coord_map_q2 = one_to_one_map_coords(pc_coords['ux'], fx_coords['ux'])
    P[pc_dofs['ux']] = fx_dofs['ux'][coord_map_q2]
    P[pc_dofs['uy']] = fx_dofs['uy'][coord_map_q2]
    coord_map_q1 = one_to_one_map_coords(pc_coords['p'], fx_coords['p'])
    P[pc_dofs['p']] = fx_dofs['p'][coord_map_q1]
    print("True DoF map discovered successfully.")
    return P

def setup_problems():
    nodes_q2, elems_q2, _, corners_q2 = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=2)
    mesh_q2 = Mesh(nodes=nodes_q2, element_connectivity=elems_q2, elements_corner_nodes=corners_q2, element_type="quad", poly_order=2)
    # nodes_q1, elems_q1, _, corners_q1 = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=1)
    # mesh_q1 = Mesh(nodes=nodes_q1, element_connectivity=elems_q1, elements_corner_nodes=corners_q1, element_type="quad", poly_order=1)
    # fe_map_pc = {'ux': mesh_q2, 'uy': mesh_q2, 'p': mesh_q1}
    mixed_element_pc = MixedElement(mesh_q2, field_specs={'ux': 2, 'uy': 2, 'p': 1})
    dof_handler_pc = DofHandler(mixed_element_pc, method='cg')
    velocity_fs = FunctionSpace("velocity", ['ux', 'uy'], dim=1)
    pressure_fs = FunctionSpace("pressure", ['p'], dim=0)
    pc = {'du': VectorTrialFunction(velocity_fs, dof_handler=dof_handler_pc), 
          'dp': TrialFunction(pressure_fs, dof_handler=dof_handler_pc), 
          'v': VectorTestFunction(velocity_fs, dof_handler=dof_handler_pc), 
          'q': TestFunction(pressure_fs, dof_handler=dof_handler_pc), 
          'u_k': VectorFunction(name="u_k", field_names=['ux', 'uy'], dof_handler=dof_handler_pc), 
          'p_k': Function(name="p_k", field_name='p', dof_handler=dof_handler_pc), 
          'u_n': VectorFunction(name="u_n", field_names=['ux', 'uy'], dof_handler=dof_handler_pc), 
          'rho': Constant(1.0,dim=0), 
          'dt': Constant(0.1,dim=0), 
          'theta': Constant(0.5,dim=0), 
          'mu': Constant(1.0e-2,dim=0)}
    
    mesh_fx = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 1, 1, dolfinx.mesh.CellType.quadrilateral)
    gdim = mesh_fx.geometry.dim
    P2_el = basix.ufl.element("Lagrange", 'quadrilateral', 2, shape=(gdim,))
    P1_el = basix.ufl.element("Lagrange", 'quadrilateral', 1)
    W_el = mixed_element([P2_el, P1_el])
    W = dolfinx.fem.functionspace(mesh_fx, W_el)
    fenicsx = {'W': W, 'rho': dolfinx.fem.Constant(mesh_fx, 1.0), 'dt': dolfinx.fem.Constant(mesh_fx, 0.1), 'theta': dolfinx.fem.Constant(mesh_fx, 0.5), 'mu': dolfinx.fem.Constant(mesh_fx, 1.0e-2)}
    V, _ = W.sub(0).collapse()
    fenicsx['u_n'] = dolfinx.fem.Function(V, name="u_n")
    fenicsx['u_k'] = dolfinx.fem.Function(W, name="u_k")
    fenicsx['c'] = dolfinx.fem.Constant(mesh_fx, (0.5, -0.2))
    
    return pc, dof_handler_pc, fenicsx

def initialize_functions(pc, fenicsx, dof_handler_pc, P_map):
    print("Initializing and synchronizing function data...")
    np.random.seed(1234)
    u_k_p_k_data_pc = np.ones(dof_handler_pc.total_dofs)  # Use ones for simplicity
    u_n_data_pc = 2*np.ones(18)  # Use ones for simplicity
    # u_k_p_k_data_pc = np.random.rand(dof_handler_pc.total_dofs)
    # u_n_data_pc = np.random.rand(18)
    # u_k_p_k_data_pc[18:] = 0.0  # Ensure pressure DoFs are zero
    
    dofs_ux_pc = dof_handler_pc.get_field_slice('ux')
    dofs_uy_pc = dof_handler_pc.get_field_slice('uy')
    dofs_p_pc = dof_handler_pc.get_field_slice('p')
    pc['u_k'].set_nodal_values(dofs_ux_pc, u_k_p_k_data_pc[dofs_ux_pc])
    pc['u_k'].set_nodal_values(dofs_uy_pc, u_k_p_k_data_pc[dofs_uy_pc])
    pc['p_k'].set_nodal_values(dofs_p_pc, u_k_p_k_data_pc[dofs_p_pc])
   
    pc['c'] = Constant([0.5,-0.2],dim=1)
    # 
    fx_u_k_array = fenicsx['u_k'].x.array
    # print(f"fx_u_k_array shape: {fx_u_k_array.shape}")
    # print(f"fx_u_k_array:{fx_u_k_array}")
    for pc_dof, fx_dof in enumerate(P_map):
    #    print(f"Mapping pc_dof {pc_dof} to fx_dof {fx_dof}")
       fx_u_k_array[fx_dof] = u_k_p_k_data_pc[pc_dof]
    
    pc['u_n'].set_nodal_values(dofs_ux_pc, u_n_data_pc[dofs_ux_pc])
    pc['u_n'].set_nodal_values(dofs_uy_pc, u_n_data_pc[dofs_uy_pc])
    
    fx_u_n_array = fenicsx['u_n'].x.array

    V, V_map = fenicsx['W'].sub(0).collapse()
    fx_global_to_local_V = np.argsort(V_map)
    pc_vel_dofs = np.concatenate([dofs_ux_pc, dofs_uy_pc])
    for i, pc_dof in enumerate(pc_vel_dofs):
        fx_global_dof = P_map[pc_dof]
        fx_local_dof_in_V = fx_global_to_local_V[fx_global_dof]
        fx_u_n_array[fx_local_dof_in_V] = u_n_data_pc[i]
    
    # print(f"before mapping, fx_u_k_array: {fx_u_k_array}")
    V, V_map = fenicsx['W'].sub(0).collapse()
    fx_global_to_local_V = np.argsort(V_map)
    pc_all_dofs = np.concatenate([dofs_ux_pc, dofs_uy_pc])
    for i, pc_dof in enumerate(pc_all_dofs):
        fx_global_dof = P_map[pc_dof]
        fx_local_dof_in_V = fx_global_to_local_V[fx_global_dof]
        fx_u_k_array[fx_local_dof_in_V] = u_k_p_k_data_pc[i]
    print(f"after mapping, fx_u_k_array: {fx_u_k_array}")

def compare_term(term_name, J_pc, R_pc, J_fx, R_fx, P_map, dof_handler_pc, W_fenicsx):
    print("\n" + f"--- Comparing Term: {term_name} ---")
    output_dir = "garbage"
    os.makedirs(output_dir, exist_ok=True)
    safe_term_name = term_name.replace(' ', '_').lower()

    if R_pc is not None and R_fx is not None:
        filename = os.path.join(output_dir, f"{safe_term_name}_residual.xlsx")
        R_fx_reordered = R_fx[P_map]
        
        # --- CORRECTION: Ensure residual vectors are 1D for DataFrame ---
        # print(R_pc.shape, R_fx_reordered.shape)
        R_pc_flat = R_pc.flatten()
        R_fx_reordered_flat = R_fx_reordered.flatten()

        pc_coords = get_all_pycutfem_dof_coords(dof_handler_pc)
        fx_coords_all = get_all_fenicsx_dof_coords(W_fenicsx)
        fx_coords_reordered = fx_coords_all[P_map]
        
        comparison_data = {
            'pc_dof_index': np.arange(dof_handler_pc.total_dofs),
            'pc_coord_x': pc_coords[:, 0], 'pc_coord_y': pc_coords[:, 1], 'pc_residual': R_pc_flat,
            'fx_coord_x': fx_coords_reordered[:, 0], 'fx_coord_y': fx_coords_reordered[:, 1],
            'fx_reordered_residual': R_fx_reordered_flat, 'abs_difference': np.abs(R_pc_flat - R_fx_reordered_flat)
        }
        pd.DataFrame(comparison_data).to_excel(filename, sheet_name='residual_comparison', index=False)
        print(f"✅ Residual comparison data saved to '{filename}'")

        try:
            np.testing.assert_allclose(R_pc_flat, R_fx_reordered_flat, rtol=1e-8, atol=1e-8)
            print(f"✅ Residual vector for '{term_name}' is numerically equivalent.")
        except AssertionError as e:
            print(f"❌ Residual vector for '{term_name}' is NOT equivalent!")
            print(e)

    if J_pc is not None and J_fx is not None:
        filename = os.path.join(output_dir, f"{safe_term_name}_jacobian.xlsx")
        J_pc_dense = J_pc.toarray()
        J_fx_reordered = J_fx[P_map, :][:, P_map]
        with pd.ExcelWriter(filename) as writer:
            pd.DataFrame(J_pc_dense).to_excel(writer, sheet_name='pycutfem', index=False, header=False)
            pd.DataFrame(J_fx_reordered).to_excel(writer, sheet_name='fenics', index=False, header=False)
        print(f"✅ Jacobian matrices saved to '{filename}'")
        try:
            np.testing.assert_allclose(J_pc_dense, J_fx_reordered, rtol=1e-8, atol=1e-8)
            print(f"✅ Jacobian matrix for '{term_name}' is numerically equivalent.")
        except AssertionError as e:
            print(f"❌ Jacobian matrix for '{term_name}' is NOT equivalent!")
            print(e)

# ==============================================================================
#                      MAIN TEST HARNESS
# ==============================================================================
if __name__ == '__main__':
    pc, dof_handler_pc, fenicsx = setup_problems()
    P_map = create_true_dof_map(dof_handler_pc, fenicsx['W'])
    initialize_functions(pc, fenicsx, dof_handler_pc, P_map)

    W_fx = fenicsx['W']
    u_k_fx, p_k_fx = ufl.split(fenicsx['u_k']) 
    u_n_fx = fenicsx['u_n']
    
    V_subspace = W_fx.sub(0)
    Q_subspace = W_fx.sub(1)
    du, v = ufl.TrialFunction(V_subspace), ufl.TestFunction(V_subspace)
    dp, q = ufl.TrialFunction(Q_subspace), ufl.TestFunction(Q_subspace)

    advection_1_pc = ( dot(dot(grad(pc['du']), pc['u_k']), pc['v'])) * dx()
    c_pc = pc['c']
    c_fx = fenicsx['c']
    jacobian_pc = (
        # Time derivative term
        pc['rho'] * dot(pc['du'], pc['v']) / pc['dt'] +
        
        # Convection terms (linearization of u ⋅ ∇u)
        pc['theta'] * pc['rho'] * dot(dot(grad(pc['u_k']), pc['du']), pc['v']) +
        pc['theta'] * pc['rho'] * dot(dot(grad(pc['du']), pc['u_k']), pc['v']) +

        # Diffusion term
        pc['theta'] * pc['mu'] * inner(grad(pc['du']), grad(pc['v'])) -
        
        # Pressure term (linearization of -p∇⋅v)
        pc['dp'] * div(pc['v']) +
        
        # Continuity term
        pc['q'] * div(pc['du'])
    ) * dx()
    residual_pc = (
        # Time derivative
        pc['rho'] * dot(pc['u_k'] - pc['u_n'], pc['v']) / pc['dt'] +

        # Convection terms (implicit and explicit parts)
        pc['theta'] * pc['rho'] * dot(dot(grad(pc['u_k']), pc['u_k']), pc['v']) +
        (1.0 - pc['theta']) * pc['rho'] * dot(dot(grad(pc['u_n']), pc['u_n']), pc['v']) +

        # Diffusion terms (implicit and explicit parts)
        pc['theta'] * pc['mu'] * inner(grad(pc['u_k']), grad(pc['v'])) +
        (1.0 - pc['theta']) * pc['mu'] * inner(grad(pc['u_n']), grad(pc['v'])) -
        
        # Pressure term
        pc['p_k'] * div(pc['v']) +
        
        # Continuity term
        pc['q'] * div(pc['u_k'])
    ) * dx()


    def create_fenics_ns_jacobian(deg):
        """Creates the UFL form for the Navier-Stokes Jacobian using Trial/Test
        functions from the mixed space W."""
        # Define Trial and Test Functions on the *mixed space* W_fx
        dup_fx, vq_fx = ufl.TrialFunction(W_fx), ufl.TestFunction(W_fx)
        # Split them to get the velocity and pressure components
        du_fx, dp_fx = ufl.split(dup_fx)
        v_fx, q_fx = ufl.split(vq_fx)
        
        # Now, build the form using these correctly-defined components
        return (
            fenicsx['rho'] * ufl.dot(du_fx, v_fx) / fenicsx['dt'] +
            fenicsx['theta'] * fenicsx['rho'] * ufl.dot(ufl.dot(ufl.grad(u_k_fx), du_fx), v_fx) +
            fenicsx['theta'] * fenicsx['rho'] * ufl.dot(ufl.dot(ufl.grad(du_fx), u_k_fx), v_fx) +
            fenicsx['theta'] * fenicsx['mu'] * ufl.inner(ufl.grad(du_fx), ufl.grad(v_fx)) -
            dp_fx * ufl.div(v_fx) +
            q_fx * ufl.div(du_fx)
        ) * ufl.dx(metadata={'quadrature_degree': deg})

    def create_fenics_ns_residual(deg):
        """Creates the UFL form for the Navier-Stokes residual using a
        TestFunction from the mixed space W."""
        # Define a single TestFunction on the parent mixed space
        vq_fx = ufl.TestFunction(W_fx)
        
        # Split it to get the velocity and pressure components
        v_fx, q_fx = ufl.split(vq_fx)
        
        # Now, build the residual form using these correctly-defined test functions
        # and the existing Function objects (u_k_fx, p_k_fx, u_n_fx)
        return (
            # Time derivative
            fenicsx['rho'] * ufl.dot(u_k_fx - u_n_fx, v_fx) / fenicsx['dt'] +

            # Convection terms (implicit and explicit parts)
            fenicsx['theta'] * fenicsx['rho'] * ufl.dot(ufl.dot(ufl.grad(u_k_fx), u_k_fx), v_fx) +
            (1.0 - fenicsx['theta']) * fenicsx['rho'] * ufl.dot(ufl.dot(ufl.grad(u_n_fx), u_n_fx), v_fx) +

            # Diffusion terms (implicit and explicit parts)
            fenicsx['theta'] * fenicsx['mu'] * ufl.inner(ufl.grad(u_k_fx), ufl.grad(v_fx)) +
            (1.0 - fenicsx['theta']) * fenicsx['mu'] * ufl.inner(ufl.grad(u_n_fx), ufl.grad(v_fx)) -
            
            # Pressure term
            p_k_fx * ufl.div(v_fx) +
            
            # Continuity term
            q_fx * ufl.div(u_k_fx)
        ) * ufl.dx(metadata={'quadrature_degree': deg})

    vq_fx = ufl.TestFunction(W_fx)
        
    # Split it to get the velocity and pressure components
    v_fx, q_fx = ufl.split(vq_fx)
    terms = {
        "LHS Mass":          {'pc': pc['rho'] * dot(pc['du'], pc['v']) / pc['dt'] * dx(),                                    'f_lambda': lambda deg: fenicsx['rho'] * ufl.dot(du, v) / fenicsx['dt'] * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "LHS Diffusion":     {'pc': pc['theta'] * pc['mu'] * inner(grad(pc['du']), grad(pc['v'])) * dx(metadata={"q":4}),                     'f_lambda': lambda deg: fenicsx['theta'] * fenicsx['mu'] * ufl.inner(ufl.grad(du), ufl.grad(v)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "LHS Advection 1":   {'pc':  advection_1_pc,           'f_lambda': lambda deg:  ufl.dot(ufl.dot(ufl.grad(du),u_k_fx), v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 5},
        "LHS Advection 2":   {'pc': pc['theta'] * pc['rho'] * dot(dot(grad(pc['u_k']), pc['du']), pc['v']) * dx(),            'f_lambda': lambda deg: fenicsx['theta'] * fenicsx['rho'] * ufl.dot(ufl.dot(ufl.grad(u_k_fx),du), v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 5},
        "LHS Pressure":      {'pc': -pc['dp'] * div(pc['v']) * dx(),                                                         'f_lambda': lambda deg: -dp * ufl.div(v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 3},
        "LHS Continuity":    {'pc': pc['q'] * div(pc['du']) * dx(),                                                          'f_lambda': lambda deg: q * ufl.div(du) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 3},
        "RHS Time Derivative": {'pc': (pc['rho'] * dot(pc['u_k'] - pc['u_n'], pc['v']) / pc['dt']) * dx(),                       'f_lambda': lambda deg: fenicsx['rho'] * ufl.dot(u_k_fx - u_n_fx, v) / fenicsx['dt'] * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 5},
        "RHS Advection":     {'pc': pc['theta'] * pc['rho'] * dot(dot(grad(pc['u_k']), pc['u_k']), pc['v']) * dx(),          'f_lambda': lambda deg: fenicsx['theta'] * fenicsx['rho'] * ufl.dot(ufl.dot(ufl.grad(u_k_fx),u_k_fx), v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 5},
        "LHS Scalar Advection": {'pc': dot(grad(pc['dp']), pc['u_k']) * pc['q'] * dx(), 'f_lambda': lambda deg: ufl.dot(ufl.grad(dp), u_k_fx) * q * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 3},
        "LHS Scalar Advection 2": {'pc': dot(pc['u_k'], grad(pc['dp'])) * pc['q'] * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.dot(u_k_fx, ufl.grad(dp)) * q * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 3},
        "LHS Vector Advection Constant": {'pc': dot(dot(grad(pc['du']), c_pc), pc['v']) * dx(), 'f_lambda': lambda deg: ufl.dot(ufl.dot(ufl.grad(du), c_fx), v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 5},
        "Navier Stokes LHS": {'pc': jacobian_pc, 'f_lambda':  create_fenics_ns_jacobian, 'mat': True, 'deg': 5},
        "RHS diffusion": {'pc': pc['theta'] * pc['mu'] * inner(grad(pc['u_k']), grad(pc['v'])) * dx(),'f_lambda': lambda deg: fenicsx['theta'] * fenicsx['mu'] * ufl.inner(ufl.grad(u_k_fx), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg':4},
        "RHS diffusion 2": {'pc': (1.0 - pc['theta']) * pc['mu'] * inner(grad(pc['u_n']), grad(pc['v'])) * dx(metadata={'quadrature_degree': 4}),'f_lambda': lambda deg: (1.0 - fenicsx['theta']) * fenicsx['mu'] * ufl.inner(ufl.grad(u_n_fx), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat':False, 'deg':4},
        "Navier Stokes RHS": {'pc': residual_pc, 'f_lambda':  create_fenics_ns_residual, 'mat': False, 'deg': 6},
        "RHS pressure term": {'pc': pc['p_k'] * div(pc['v']) * dx, 'f_lambda': lambda deg: p_k_fx * ufl.div(v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 5},
        "RHS Continuity":    {'pc': pc['q'] * div(pc['u_k']) * dx, 'f_lambda': lambda deg: q_fx * ufl.div(u_k_fx) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 6},
        "distributed rhs":    {'pc': -(pc['q'] * div(pc['u_k']) * dx - pc['p_k'] * div(pc['v']) * dx), 'f_lambda': lambda deg: -(q_fx * ufl.div(u_k_fx) * ufl.dx(metadata={'quadrature_degree': deg}) - p_k_fx * ufl.div(v_fx) * ufl.dx(metadata={'quadrature_degree': deg})), 'mat': False, 'deg': 6}

    }
    pc_dummy_side = dot(Constant([0.0,0.0],dim=1), pc['v']) * dx()
    for name, forms in terms.items():
        J_pc, R_pc, J_fx, R_fx = None, None, None, None
        
        form_fx_ufl = forms['f_lambda'](forms['deg'])
        form_fx_compiled = dolfinx.fem.form(form_fx_ufl)
        print(f"Compiling form for '{name}' with degree {forms['deg']}...")

        if forms['mat']:
            J_pc, _ = assemble_form(forms['pc'] == pc_dummy_side, dof_handler_pc, quad_degree=forms['deg'])
            A = dolfinx.fem.petsc.assemble_matrix(form_fx_compiled)
            A.assemble()
            indptr, indices, data = A.getValuesCSR()
            J_fx_sparse = csr_matrix((data, indices, indptr), shape=A.getSize())
            J_fx = J_fx_sparse.toarray()
        else:
            _, R_pc = assemble_form( pc_dummy_side==forms['pc'], dof_handler_pc) # forms['deg']
            vec = dolfinx.fem.petsc.assemble_vector(form_fx_compiled)
            # CORRECTED: ghostUpdate is not needed for serial runs.
            R_fx = vec.array
        
        compare_term(name, J_pc, R_pc, J_fx, R_fx, P_map, dof_handler_pc, W_fx)