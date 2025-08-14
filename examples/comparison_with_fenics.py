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
    Function, VectorFunction, Constant, grad, inner, dot, div, trace, Hessian, Laplacian
)
from pycutfem.ufl.measures import dx, dInterface
from pycutfem.ufl.forms import assemble_form
from pycutfem.fem.reference import get_reference
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.forms import Equation

# Imports for mapping and matrix conversion
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
import logging
logging.basicConfig(
    level=logging.INFO,  # show debug messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def debug_interpolate(self, f):
    """
    Calculates direct nodal interpolation values for a function space.
    This function ONLY calculates and returns the values; it does not modify the Function object.
    """
    print(f"--- Calculating values for {self.name} ---")
    fs = self.function_space
    try:
        x_dofs = fs.tabulate_dof_coordinates()
    except RuntimeError:
        print(f"Space with element '{type(fs.ufl_element()).__name__}' is a subspace. Collapsing...")
        collapsed_space, _ = fs.collapse()
        x_dofs = collapsed_space.tabulate_dof_coordinates()
    
    if x_dofs.shape[0] == 0:
        return np.array([], dtype=np.float64)

    values = np.asarray(f(x_dofs.T))
    if len(values.shape) == 1:
        values = values.reshape(1, -1)
    
    # Return the flattened, interleaved array of nodal values
    return values.T.flatten()

# MONKEY-PATCH: Add our new method to the dolfinx.fem.Function class
dolfinx.fem.Function.debug_interpolate = debug_interpolate
# Helper functions for coordinates
def get_pycutfem_dof_coords(dof_handler: DofHandler, field: str) -> np.ndarray:
    if field not in dof_handler.field_names:
        raise ValueError(f"Field '{field}' not found in DofHandler")
    return dof_handler.get_dof_coords(field)

def get_all_pycutfem_dof_coords(dof_handler: DofHandler) -> np.ndarray:
    all_coords = np.zeros((dof_handler.total_dofs, 2))
    for field in ['ux', 'uy', 'p']:
        field_dofs = dof_handler.get_field_slice(field)
        field_coords = get_pycutfem_dof_coords(dof_handler, field)
        all_coords[field_dofs] = field_coords
    return all_coords

def get_all_fenicsx_dof_coords(W_fenicsx):
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
    nx, ny = 1, 1  # Number of elements in x and y directions
    nodes_q2, elems_q2, _, corners_q2 = structured_quad(1.0, 1.0, nx=nx, ny=ny, poly_order=2)
    mesh_q2 = Mesh(nodes=nodes_q2, element_connectivity=elems_q2, elements_corner_nodes=corners_q2, element_type="quad", poly_order=2)
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
    
    mesh_fx = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, dolfinx.mesh.CellType.quadrilateral)
    gdim = mesh_fx.geometry.dim
    P2_el = basix.ufl.element("Lagrange", 'quadrilateral', 2, shape=(gdim,))
    P1_el = basix.ufl.element("Lagrange", 'quadrilateral', 1)
    W_el = mixed_element([P2_el, P1_el])
    W = dolfinx.fem.functionspace(mesh_fx, W_el)
    fenicsx = {'W': W, 'rho': dolfinx.fem.Constant(mesh_fx, 1.0), 'dt': dolfinx.fem.Constant(mesh_fx, 0.1), 'theta': dolfinx.fem.Constant(mesh_fx, 0.5), 'mu': dolfinx.fem.Constant(mesh_fx, 1.0e-2)}
    V, _ = W.sub(0).collapse()
    fenicsx['u_n'] = dolfinx.fem.Function(V, name="u_n")
    fenicsx['u_k_p_k'] = dolfinx.fem.Function(W, name="u_k_p_k")
    fenicsx['c'] = dolfinx.fem.Constant(mesh_fx, (0.5, -0.2))
    
    return pc, dof_handler_pc, fenicsx

def initialize_functions(pc, fenicsx, dof_handler_pc, P_map):
    print("Initializing and synchronizing function data...")
    np.random.seed(1234)
    def ones(x):
        return np.ones_like(x)
    def u_k_init_func(x):
        # return [np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]) , -np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]) ]
        # return [ x[0] *  x[1]**2 , -x[0] *  x[1] ]
        return [11 + x[0]  * x[1], 33 + x[1]**2]
        # return [x[0]**2 * x[1], x[1]**2]
        # return [ones(x[0]), ones(x[1])]
    def p_k_init_func(x):
        return np.sin(2 * np.pi * x[0]*x[1])
        # return ones(x[0]) 
    def u_n_init_func(x):
        return [0.5 * val for val in u_k_init_func(x)]
        # return [0.5 * ones(x[0]), 0.5 * ones(x[1])]

    # u_k_p_k_data_pc[3:5] = 8.0
    
    # --- Initialize pycutfem ---
    pc['u_k'].set_values_from_function(lambda x, y: u_k_init_func([x, y]))
    pc['p_k'].set_values_from_function(lambda x, y: p_k_init_func([x, y]))
    pc['u_n'].set_values_from_function(lambda x, y: u_n_init_func([x, y]))
    pc['c'] = Constant([0.5,-0.2],dim=1)
    
    # --- Initialize FEniCSx and Verify ---
    W = fenicsx['W']
    u_k_p_k_fx = fenicsx['u_k_p_k']
    u_n_fx = fenicsx['u_n']
    
    # Get the maps from the subspaces (V, Q) to the parent space (W)
    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()

    # Get the component "views"
    u_k_fx = u_k_p_k_fx.sub(0)
    p_k_fx = u_k_p_k_fx.sub(1)

    # --- This is the corrected assignment logic ---
    # 1. Calculate the values for the velocity subspace
    u_k_values = u_k_fx.debug_interpolate(u_k_init_func)
    # 2. Place them into the correct slots of the PARENT vector using the map
    u_k_p_k_fx.x.array[V_to_W] = u_k_values

    # 3. Calculate the values for the pressure subspace
    p_k_values = p_k_fx.debug_interpolate(p_k_init_func)
    # 4. Place them into the correct slots of the PARENT vector using the map
    u_k_p_k_fx.x.array[Q_to_W] = p_k_values
    
    # 5. Synchronize the PARENT vector once after all modifications
    u_k_p_k_fx.x.scatter_forward()
    
    # 6. Initialize the standalone u_n function (this needs its own assignment and scatter)
    u_n_values = u_n_fx.debug_interpolate(u_n_init_func)
    u_n_fx.x.array[:] = u_n_values
    u_n_fx.x.scatter_forward()

    # --- Optional Assertion ---
    pycutfem_uk_values = pc['u_k'].nodal_values
    np.testing.assert_allclose(np.sort(pycutfem_uk_values), np.sort(u_k_values), rtol=1e-8, atol=1e-15)
    print("\n✅ ASSERTION PASSED: pycutfem and FEniCSx calculated the same set of nodal values for u_k.")

def compare_term(term_name, J_pc, R_pc, J_fx, R_fx, P_map, dof_handler_pc, W_fenicsx):
    print("\n" + f"--- Comparing Term: {term_name} ---")

    output_dir = "garbage"
    os.makedirs(output_dir, exist_ok=True)
    safe_term_name = term_name.replace(' ', '_').lower()
    is_successful = True

    if R_pc is not None and R_fx is not None:
        filename = os.path.join(output_dir, f"{safe_term_name}_residual.xlsx")
        R_fx_reordered = R_fx[P_map]
        
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
            is_successful = False

    if J_pc is not None and J_fx is not None:
        filename = os.path.join(output_dir, f"{safe_term_name}_jacobian.xlsx")
        J_pc_dense = J_pc.toarray()
        J_fx_reordered = J_fx[P_map, :][:, P_map]
        with pd.ExcelWriter(filename) as writer:
            pd.DataFrame(J_pc_dense).to_excel(writer, sheet_name='pycutfem', index=False, header=False)
            pd.DataFrame(J_fx_reordered).to_excel(writer, sheet_name='fenics', index=False, header=False)
            pd.DataFrame(np.abs(J_pc_dense - J_fx_reordered)<1e-12).to_excel(writer, sheet_name='difference', index=False, header=False)
        print(f"✅ Jacobian matrices saved to '{filename}'")
        try:
            np.testing.assert_allclose(J_pc_dense, J_fx_reordered, rtol=1e-8, atol=1e-8)
            print(f"✅ Jacobian matrix for '{term_name}' is numerically equivalent.")
        except AssertionError as e:
            print(f"❌ Jacobian matrix for '{term_name}' is NOT equivalent!")
            print(e)
            is_successful = False
    return is_successful

def print_test_summary(success_count, failed_tests):
    """Prints a summary of the test results."""
    total_tests = success_count + len(failed_tests)
    failure_count = len(failed_tests)

    print("\n" + "="*70)
    print(" " * 25 + "TEST SUITE SUMMARY")
    print("="*70)
    print(f"Total tests run: {total_tests}")
    print(f"✅ Successful tests: {success_count}")
    print(f"❌ Failed tests:     {failure_count}")
    
    if failure_count > 0:
        print("\n--- List of Failed Tests ---")
        for test_name in failed_tests:
            print(f"  - {test_name}")
    print("="*70)

# ==============================================================================
#                      MAIN TEST HARNESS
# ==============================================================================
if __name__ == '__main__':
    pc, dof_handler_pc, fenicsx = setup_problems()
    P_map = create_true_dof_map(dof_handler_pc, fenicsx['W'])
    initialize_functions(pc, fenicsx, dof_handler_pc, P_map)

    W_fx = fenicsx['W']
    u_k_fx, p_k_fx = ufl.split(fenicsx['u_k_p_k']) 
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

    def epsilon_f(u, grad):
        "Symmetric gradient."
        return 0.5 * (grad(u) + grad(u).T)

    def epsilon_s_linear_L(disp, disp_k, grad, dot):
        """Agnostic version of the linearized solid strain tensor (LHS)."""
        # return 0.5 * (grad(disp) + grad(disp).T +  dot(grad(disp).T, grad(disp_k))  + dot(grad(disp_k).T, grad(disp)))
        return 0.5 * ( dot(grad(disp_k).T, grad(disp)))

    def epsilon_s_linear_R(disp_k, grad, dot):
        """Agnostic version of the linearized solid strain tensor (RHS)."""
        return 0.5 * (grad(disp_k) + grad(disp_k).T + dot(grad(disp_k).T, grad(disp_k)))

    def trace_component(disp, disp_k, grad, dot, trace):
        return 0.5 * ( trace(grad(disp)) + trace(grad(disp).T) + trace(dot(grad(disp).T, grad(disp_k))) + trace(dot(grad(disp_k).T, grad(disp))))
    def sigma_s_linear_weak_L(ddisp, disp_k, grad_v_test, inner, trace, grad, dot):
        """Agnostic version of the linearized solid stress tensor (LHS)."""
        strain = epsilon_s_linear_L(ddisp, disp_k, grad, dot)
        return   trace(strain) * trace(grad_v_test) #+ 2.0 * inner(strain, grad_v_test)
        # return 2.0 * inner(strain, grad_v_test) + trace_component(ddisp, disp_k, grad, dot, trace) * trace(grad_v_test)

    def sigma_s_linear_weak_R(disp_k, grad_v_test, inner, trace, grad, dot):
        """Agnostic version of the linearized solid stress tensor (RHS)."""
        strain = epsilon_s_linear_R(disp_k, grad, dot)
        return 2.0 * inner(strain, grad_v_test) + trace(strain) * trace(grad_v_test)

    terms = {
        # "LHS Mass":          {'pc': pc['rho'] * dot(pc['du'], pc['v']) / pc['dt'] * dx(),                                    'f_lambda': lambda deg: fenicsx['rho'] * ufl.dot(du, v) / fenicsx['dt'] * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        # "LHS Diffusion":     {'pc': pc['theta'] * pc['mu'] * inner(grad(pc['du']), grad(pc['v'])) * dx(metadata={"q":4}),                     'f_lambda': lambda deg: fenicsx['theta'] * fenicsx['mu'] * ufl.inner(ufl.grad(du), ufl.grad(v)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        # "LHS Advection 1":   {'pc':  ( dot(dot(grad(pc['du']), pc['u_k']), pc['v'])) * dx(metadata={"q":6}),           'f_lambda': lambda deg:  ufl.dot(ufl.dot(ufl.grad(du),u_k_fx), v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 5},
        # "LHS Advection 2":   {'pc':  dot(dot(grad(pc['u_k']), pc['du']), pc['v']) * dx(metadata={"q":5}),            'f_lambda': lambda deg: ufl.dot( ufl.dot(ufl.grad(u_k_fx),du), v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 5},
        # "LHS Advection 3":   {'pc':  ( dot(dot(pc['u_k'], grad(pc['du']) ), pc['v'])) * dx(metadata={"q":5}),           'f_lambda': lambda deg:  ufl.dot(ufl.dot(u_k_fx,ufl.nabla_grad(du)), v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 5},
        # "LHS Advection 4":   {'pc': pc['theta'] * pc['rho'] * dot(dot(pc['du'],grad(pc['u_k']) ), pc['v']) * dx(metadata={"q":5}),            'f_lambda': lambda deg: fenicsx['theta'] * fenicsx['rho'] * ufl.dot(ufl.dot(du,ufl.nabla_grad(u_k_fx)), v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 5},
        # "LHS Pressure":      {'pc': -pc['dp'] * div(pc['v']) * dx(),                                                         'f_lambda': lambda deg: -dp * ufl.div(v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 3},
        # "LHS Continuity":    {'pc': pc['q'] * div(pc['du']) * dx(),                                                          'f_lambda': lambda deg: q * ufl.div(du) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 3},
        # "RHS Time Derivative": {'pc': (pc['rho'] * dot(pc['u_k'] - pc['u_n'], pc['v']) / pc['dt']) * dx(),                       'f_lambda': lambda deg: fenicsx['rho'] * ufl.dot(u_k_fx - u_n_fx, v) / fenicsx['dt'] * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 5},
        # "RHS Advection":     {'pc': pc['theta'] * pc['rho'] * dot(dot(grad(pc['u_k']), pc['u_k']), pc['v']) * dx(),          'f_lambda': lambda deg: fenicsx['theta'] * fenicsx['rho'] * ufl.dot(ufl.dot(ufl.grad(u_k_fx),u_k_fx), v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 5},
        # "RHS Advection 2":     {'pc': pc['theta'] * pc['rho'] * dot(dot(pc['u_k'],grad(pc['u_k']) ), pc['v']) * dx(),          'f_lambda': lambda deg: fenicsx['theta'] * fenicsx['rho'] * ufl.dot(ufl.dot(u_k_fx,ufl.nabla_grad(u_k_fx)), v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 5},
        # "LHS Scalar Advection": {'pc': dot(grad(pc['dp']), pc['u_k']) * pc['q'] * dx(), 'f_lambda': lambda deg: ufl.dot(ufl.grad(dp), u_k_fx) * q * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 3},
        # "LHS Scalar Advection 2": {'pc': dot(pc['u_k'], grad(pc['dp'])) * pc['q'] * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.dot(u_k_fx, ufl.grad(dp)) * q * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 3},
        # "LHS Vector Advection Constant": {'pc': dot(dot(grad(pc['du']), c_pc), pc['v']) * dx(), 'f_lambda': lambda deg: ufl.dot(ufl.dot(ufl.grad(du), c_fx), v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 5},
        # "Navier Stokes LHS": {'pc': jacobian_pc, 'f_lambda':  create_fenics_ns_jacobian, 'mat': True, 'deg': 5},
        # "RHS diffusion": {'pc': inner(grad(pc['u_k']), grad(pc['v'])) * dx(metadata={"q":4}),'f_lambda': lambda deg:  ufl.inner(ufl.grad(u_k_fx), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg':4},
        # "RHS scalar diffusion": {'pc':  inner(grad(pc['p_k']), grad(pc['q'])) * dx(metadata={"q":4}),'f_lambda': lambda deg: ufl.inner(ufl.grad(p_k_fx), ufl.grad(q_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg':4},
        # "RHS diffusion 3": {'pc': (1.0 - pc['theta']) * pc['mu'] * inner(grad(pc['u_n']), grad(pc['v'])) * dx(metadata={'quadrature_degree': 4}),'f_lambda': lambda deg: (1.0 - fenicsx['theta']) * fenicsx['mu'] * ufl.inner(ufl.grad(u_n_fx), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat':False, 'deg':4},
        # "Navier Stokes RHS": {'pc': residual_pc, 'f_lambda':  create_fenics_ns_residual, 'mat': False, 'deg': 6},
        # "RHS pressure term": {'pc': pc['p_k'] * div(pc['v']) * dx, 'f_lambda': lambda deg: p_k_fx * ufl.div(v) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 5},
        # "RHS Continuity":    {'pc': pc['q'] * div(pc['u_k']) * dx, 'f_lambda': lambda deg: q_fx * ufl.div(u_k_fx) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 6},# "distributed rhs": {'pc': -(pc['q'] * div(pc['u_k']) * dx - pc['p_k'] * div(pc['v']) * dx), 'f_lambda': lambda deg: -(q_fx * ufl.div(u_k_fx) * ufl.dx(metadata={'quadrature_degree': deg}) - p_k_fx * ufl.div(v_fx) * ufl.dx(metadata={'quadrature_degree': deg})), 'mat': False, 'deg': 6}
        # "Dot of gradients LHS ohne transpose": {'pc':  inner(dot(grad(pc['du']), grad(pc['u_k'])),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(du), ufl.grad(u_k_fx)), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        # "Dot of gradients LHS  transpose": {'pc':  inner(dot(grad(pc['du']).T, grad(pc['u_k'])),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(du).T, ufl.grad(u_k_fx)), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        # "Dot of gradients LHS  transpose 2": {'pc':  inner(dot(grad(pc['du']), grad(pc['u_k']).T),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(du), ufl.grad(u_k_fx).T), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        # "Dot of gradients LHS  transpose 3": {'pc':  inner(dot(grad(pc['du']).T, grad(pc['u_k']).T),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(du).T, ufl.grad(u_k_fx).T), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        # "Dot of gradients LHS swap ohne transpose": {'pc':  inner(dot(grad(pc['u_k']),grad(pc['du']) ),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx),ufl.grad(du) ), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        # "Dot of gradients LHS swap  transpose 1": {'pc':  inner(dot(grad(pc['u_k']),grad(pc['du']).T),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx), ufl.grad(du).T), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        # "Dot of gradients LHS swap transpose 2": {'pc':  inner(dot( grad(pc['u_k']).T, grad(pc['du'])),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot( ufl.grad(u_k_fx).T , ufl.grad(du)), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        # "Dot of gradients LHS swap transpose 3": {'pc':  inner(dot( grad(pc['u_k']).T, grad(pc['du']).T),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot( ufl.grad(u_k_fx).T , ufl.grad(du).T), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        # "Dot of gradients RHS ohne transpose": {'pc':  inner(dot(grad(pc['u_k']), grad(pc['u_k'])),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx), ufl.grad(u_k_fx)), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        # "Dot of gradients RHS transpose": {'pc':  inner(dot(grad(pc['u_k']).T, grad(pc['u_k'])),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx).T,  ufl.grad(u_k_fx)), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        # "Dot of gradients RHS transpose 2": {'pc':  inner(dot(grad(pc['u_k']), grad(pc['u_k']).T),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx),  ufl.grad(u_k_fx).T), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        # "Dot of gradients RHS transpose 3": {'pc':  inner(dot(grad(pc['u_k']).T, grad(pc['u_k']).T),grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx).T,  ufl.grad(u_k_fx).T), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        # "LHS Diffusion transpose 1":     {'pc': pc['theta'] * pc['mu'] * inner(grad(pc['du']), grad(pc['v']).T) * dx(metadata={"q":4}),                     'f_lambda': lambda deg: fenicsx['theta'] * fenicsx['mu'] * ufl.inner(ufl.grad(du), ufl.grad(v).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        # "LHS Diffusion transpose 2":     {'pc': pc['theta'] * pc['mu'] * inner(grad(pc['du']).T, grad(pc['v'])) * dx(metadata={"q":4}),                     'f_lambda': lambda deg: fenicsx['theta'] * fenicsx['mu'] * ufl.inner(ufl.grad(du).T, ufl.grad(v)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        # "LHS Diffusion transpose 3":     {'pc': pc['theta'] * pc['mu'] * inner(grad(pc['du']).T, grad(pc['v']).T) * dx(metadata={"q":4}),                     'f_lambda': lambda deg: fenicsx['theta'] * fenicsx['mu'] * ufl.inner(ufl.grad(du).T, ufl.grad(v).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        # "LHS Diffusion transpose 4":     {'pc': pc['theta'] * pc['mu'] * inner(grad(pc['du']) + grad(pc['du']).T, grad(pc['v']) + grad(pc['v']).T) * dx(metadata={"q":4}),                     'f_lambda': lambda deg: fenicsx['theta'] * fenicsx['mu'] * ufl.inner(ufl.grad(du) + ufl.grad(du).T, ufl.grad(v) + ufl.grad(v).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        # "Dot 2 of gradients LHS ohne transpose": {'pc':  inner(dot(grad(pc['du']), grad(pc['u_k'])),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(du), ufl.grad(u_k_fx)), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        # "Dot 2 of gradients LHS  transpose": {'pc':  inner(dot(grad(pc['du']).T, grad(pc['u_k'])),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(du).T, ufl.grad(u_k_fx)), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        # "Dot 2 of gradients LHS  transpose 2": {'pc':  inner(dot(grad(pc['du']), grad(pc['u_k']).T),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(du), ufl.grad(u_k_fx).T), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        # "Dot 2 of gradients LHS  transpose 3": {'pc':  inner(dot(grad(pc['du']).T, grad(pc['u_k']).T),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(du).T, ufl.grad(u_k_fx).T), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        # "Dot 2 of gradients LHS swap ohne transpose": {'pc':  inner(dot(grad(pc['u_k']),grad(pc['du']) ),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx),ufl.grad(du) ), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        # "Dot 2 of gradients LHS swap  transpose 1": {'pc':  inner(dot(grad(pc['u_k']),grad(pc['du']).T),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx), ufl.grad(du).T), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        # "Dot 2 of gradients LHS swap transpose 2": {'pc':  inner(dot( grad(pc['u_k']).T, grad(pc['du'])),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot( ufl.grad(u_k_fx).T , ufl.grad(du)), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        # "Dot 2 of gradients LHS swap transpose 3": {'pc':  inner(dot( grad(pc['u_k']).T, grad(pc['du']).T),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot( ufl.grad(u_k_fx).T , ufl.grad(du).T), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        # "Dot 2 of gradients RHS ohne transpose": {'pc':  inner(dot(grad(pc['u_k']), grad(pc['u_k'])),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx), ufl.grad(u_k_fx)), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        # "Dot 2 of gradients RHS transpose": {'pc':  inner(dot(grad(pc['u_k']).T, grad(pc['u_k'])),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx).T,  ufl.grad(u_k_fx)), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        # "Dot 2 of gradients RHS transpose 2": {'pc':  inner(dot(grad(pc['u_k']), grad(pc['u_k']).T),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx),  ufl.grad(u_k_fx).T), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        # "Dot 2 of gradients RHS transpose 3": {'pc':  inner(dot(grad(pc['u_k']).T, grad(pc['u_k']).T),grad(pc['v']).T) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.dot(ufl.grad(u_k_fx).T,  ufl.grad(u_k_fx).T), ufl.grad(v_fx).T) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        
        # "Linear Green Stress": {'pc':  sigma_s_linear_weak_L(pc['du'], pc['u_k'], grad(pc['v']),inner, trace, grad, dot) * dx(metadata={"q":6}), 'f_lambda': lambda deg: sigma_s_linear_weak_L(du,u_k_fx, ufl.grad(v_fx), ufl.inner, ufl.tr, ufl.grad, ufl.dot) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 6},
        # "Linearized Strain LHS": {'pc': inner(epsilon_s_linear_L(pc['du'], pc['u_k'], grad, dot), grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(epsilon_s_linear_L(du, u_k_fx, ufl.grad, ufl.dot), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},

        # "Linearized Strain RHS": {'pc': inner(epsilon_s_linear_R(pc['u_k'], grad, dot), grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(epsilon_s_linear_R(u_k_fx, ufl.grad, ufl.dot), ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},

        # "Linear Green Stress RHS": {'pc': sigma_s_linear_weak_R(pc['u_k'], grad(pc['v']), inner, trace, grad, dot) * dx(metadata={"q":4}), 'f_lambda': lambda deg: sigma_s_linear_weak_R(u_k_fx, ufl.grad(v_fx), ufl.inner, ufl.tr, ufl.grad, ufl.dot) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},

        # "Trace Operator LHS": {'pc': trace(grad(pc['du'])) * trace(grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.tr(ufl.grad(du)) * ufl.tr(ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},

        # "Trace Operator RHS": {'pc': trace(grad(pc['u_k'])) * trace(grad(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.tr(ufl.grad(u_k_fx)) * ufl.tr(ufl.grad(v_fx)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        # "Linear Strain tensor LHS": {'pc': inner(epsilon_f(pc['du'], grad), epsilon_f(pc['v'],grad))  * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(epsilon_f(du, ufl.grad), epsilon_f(v_fx, ufl.grad)) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Vector Hessian LHS": {'pc': inner(Hessian(pc['du']), Hessian(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.grad(ufl.grad(du)), ufl.grad(ufl.grad(v_fx))) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Vector Hessian RHS": {'pc': inner(Hessian(pc['u_k']), Hessian(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.grad(ufl.grad(u_k_fx)), ufl.grad(ufl.grad(v_fx))) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        "Vector Laplacian LHS": {'pc': inner(Laplacian(pc['du']), Laplacian(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.div(ufl.grad(du)), ufl.div(ufl.grad(v_fx))) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Vector Laplacian RHS": {'pc': inner(Laplacian(pc['u_k']), Laplacian(pc['v'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.div(ufl.grad(u_k_fx)), ufl.div(ufl.grad(v_fx))) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        "Scalar Hessian LHS": {'pc': inner(Hessian(pc['dp']), Hessian(pc['q'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.grad(ufl.grad(dp)), ufl.grad(ufl.grad(q_fx))) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Scalar Hessian RHS": {'pc': inner(Hessian(pc['p_k']), Hessian(pc['q'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.grad(ufl.grad(p_k_fx)), ufl.grad(ufl.grad(q_fx))) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},
        "Scalar Laplacian LHS": {'pc': inner(Laplacian(pc['dp']), Laplacian(pc['q'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.div(ufl.grad(dp)), ufl.div(ufl.grad(q_fx))) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': True, 'deg': 4},
        "Scalar Laplacian RHS": {'pc': inner(Laplacian(pc['p_k']), Laplacian(pc['q'])) * dx(metadata={"q":4}), 'f_lambda': lambda deg: ufl.inner(ufl.div(ufl.grad(p_k_fx)), ufl.div(ufl.grad(q_fx))) * ufl.dx(metadata={'quadrature_degree': deg}), 'mat': False, 'deg': 4},

        }
    pc_dummy_side = dot(Constant([0.0,0.0],dim=1), pc['v']) * dx()
    # Keep track of successes and failures
    failed_tests = []
    success_count = 0
    backend_type = "jit"
    for name, forms in terms.items():
        J_pc, R_pc, J_fx, R_fx = None, None, None, None

        form_fx_ufl = forms['f_lambda'](forms['deg'])
        form_fx_compiled = dolfinx.fem.form(form_fx_ufl)
        print(f"Compiling form for '{name}' with degree {forms['deg']}...")

        if forms['mat']:
            J_pc, _ = assemble_form(Equation(forms['pc'], None), dof_handler_pc, quad_degree=forms['deg'], bcs=[], 
                                    backend=backend_type)
            A = dolfinx.fem.petsc.assemble_matrix(form_fx_compiled)
            A.assemble()
            indptr, indices, data = A.getValuesCSR()
            J_fx_sparse = csr_matrix((data, indices, indptr), shape=A.getSize())
            J_fx = J_fx_sparse.toarray()
        else:
            _, R_pc = assemble_form(Equation(None, forms['pc']), dof_handler_pc, bcs=[], 
                                    backend=backend_type)
            vec = dolfinx.fem.petsc.assemble_vector(form_fx_compiled)
            # CORRECTED: ghostUpdate is not needed for serial runs.
            R_fx = vec.array

        is_success = compare_term(name, J_pc, R_pc, J_fx, R_fx, P_map, dof_handler_pc, W_fx)
        if is_success:
            success_count += 1
        else:
            failed_tests.append(name)
    # Print the final summary of all tests
    print_test_summary(success_count, failed_tests)