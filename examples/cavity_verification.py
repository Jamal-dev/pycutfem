import numpy as np
import time
import os
import scipy.sparse.linalg as sp_la
import matplotlib.pyplot as plt

# --- Core imports ---
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad

# --- UFL-like imports ---
from pycutfem.ufl.functionspace import FunctionSpace
from pycutfem.ufl.expressions import (
    TrialFunction, TestFunction, VectorTrialFunction, VectorTestFunction,
    Function, VectorFunction, Constant, grad, inner, dot, div
)
from pycutfem.ufl.measures import dx
from pycutfem.ufl.forms import BoundaryCondition, assemble_form
from pycutfem.fem.mixedelement import MixedElement


# --- NEW: High-performance backend imports ---
from pycutfem.jit import compile_backend

# ============================================================================
#    NEW: High-Performance Assembly Helper
# ============================================================================
def assemble_system_from_local(K_loc, F_loc, dof_handler, bcs):
    """
    Assembles global sparse matrix and vector from local contributions
    and applies boundary conditions.
    """
    n_total_dofs = dof_handler.total_dofs
    n_elements, n_dofs_local, _ = K_loc.shape
    
    # Pre-allocate COO data
    data = np.zeros(n_elements * n_dofs_local * n_dofs_local)
    rows = np.zeros_like(data, dtype=np.int32)
    cols = np.zeros_like(data, dtype=np.int32)
    F = np.zeros(n_total_dofs, dtype=np.float64)

    # Build COO triplets and global vector F
    for e in range(n_elements):
        gdofs = dof_handler.get_elemental_dofs(e)
        r, c = np.meshgrid(gdofs, gdofs, indexing='ij')
        
        start = e * n_dofs_local * n_dofs_local
        end = start + n_dofs_local * n_dofs_local
        
        rows[start:end] = r.ravel()
        cols[start:end] = c.ravel()
        data[start:end] = K_loc[e].ravel()
        np.add.at(F, gdofs, F_loc[e])

    K = sp_la.coo_matrix((data, (rows, cols)), shape=(n_total_dofs, n_total_dofs)).tocsr()
    
    # Apply Dirichlet boundary conditions
    if bcs:
        bc_data = dof_handler.get_dirichlet_data(bcs)
        if bc_data:
            bc_dofs = np.fromiter(bc_data.keys(), dtype=int)
            bc_vals = np.fromiter(bc_data.values(), dtype=float)
            
            F -= K @ np.bincount(bc_dofs, weights=bc_vals, minlength=n_total_dofs)
            
            K_lil = K.tolil()
            K_lil[bc_dofs, :] = 0
            K_lil[:, bc_dofs] = 0
            K_lil[bc_dofs, bc_dofs] = 1.0
            K = K_lil.tocsr()
            
            F[bc_dofs] = bc_vals
            
    return K, F

# ============================================================================
#    Verification Data and Plotting Function
# ============================================================================
ghia_data_re100 = {
    'y_locations': np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]),
    'u_velocity_on_vertical_centerline': np.array([0.0, -0.0722, -0.1364, -0.2282, -0.2928, -0.3239, -0.3273, -0.3017, -0.2452, -0.1553, -0.0524, 0.0033, 1.0]),
    'x_locations': np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]),
    'v_velocity_on_horizontal_centerline': np.array([0.0, 0.0886, 0.1608, 0.2804, 0.3556, 0.3789, 0.3547, 0.2971, 0.2223, 0.1463, 0.0712, 0.0396, 0.0])
}

def create_verification_plot(dof_handler, u_solution, reference_data):
    """
    Extracts velocity profiles, plots them against Ghia et al. reference data,
    and saves the plot to a file.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Verification Against Ghia, Ghia & Shin (1982) for Re=100", fontsize=16)

    # --- 1. u-velocity on vertical centerline (x=0.5) ---
    ux_dof_coords = dof_handler.get_dof_coords('ux')
    centerline_mask = np.isclose(ux_dof_coords[:, 0], 0.5)
    y_coords = ux_dof_coords[centerline_mask, 1]
    u_centerline = u_solution[0].nodal_values[centerline_mask]
    sort_indices = np.argsort(y_coords)
    ax = axes[0]
    ax.plot(u_centerline[sort_indices], y_coords[sort_indices], 'b-', label='FEM Solution', lw=2)
    ax.plot(reference_data['u_velocity_on_vertical_centerline'], reference_data['y_locations'], 'ro', label='Ghia et al. (1982)', mfc='none')
    ax.set_title('u-velocity along Vertical Centerline (x=0.5)')
    ax.set_xlabel('u-velocity')
    ax.set_ylabel('Y-coordinate')
    ax.grid(True, linestyle=':')
    ax.legend()

    # --- 2. v-velocity on horizontal centerline (y=0.5) ---
    uy_dof_coords = dof_handler.get_dof_coords('uy')
    centerline_mask = np.isclose(uy_dof_coords[:, 1], 0.5)
    x_coords = uy_dof_coords[centerline_mask, 0]
    v_centerline = u_solution[1].nodal_values[centerline_mask]
    sort_indices = np.argsort(x_coords)
    ax = axes[1]
    ax.plot(x_coords[sort_indices], v_centerline[sort_indices], 'b-', label='FEM Solution', lw=2)
    ax.plot(reference_data['x_locations'], reference_data['v_velocity_on_horizontal_centerline'], 'ro', label='Ghia et al. (1982)', mfc='none')
    ax.set_title('v-velocity along Horizontal Centerline (y=0.5)')
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('v-velocity')
    ax.grid(True, linestyle=':')
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # NEW: Save the plot to a file
    plt.savefig("cavity_verification_plot.png", dpi=300)
    print("Saved verification plot to cavity_verification_plot.png")
    plt.close(fig)

# 1. ============================================================================
#    SETUP
# ===============================================================================
L, H = 1.0, 1.0
NX, NY = 32, 32
nodes_q2, elems_q2, _, corners_q2 = structured_quad(L, H, nx=NX, ny=NY, poly_order=2)
mesh_q2 = Mesh(nodes=nodes_q2, element_connectivity=elems_q2, elements_corner_nodes=corners_q2, element_type="quad", poly_order=2)
mixed_element = MixedElement(mesh_q2, field_specs={'ux': 2, 'uy': 2, 'p': 1})
dof_handler = DofHandler(mixed_element, method='cg')

bc_tags = {'bottom_wall': lambda x,y: np.isclose(y,0), 'left_wall': lambda x,y: np.isclose(x,0), 'right_wall': lambda x,y: np.isclose(x,L), 'top_lid': lambda x,y: np.isclose(y,H)}
mesh_q2.tag_boundary_edges(bc_tags)
dof_handler.tag_dof_by_locator(tag='pressure_pin_point', field='p', locator=lambda x, y: np.isclose(x, 0.0) and np.isclose(y, 0.0), find_first=True)

bcs = [
    BoundaryCondition('ux', 'dirichlet', 'bottom_wall', lambda x,y: 0.0), BoundaryCondition('uy', 'dirichlet', 'bottom_wall', lambda x,y: 0.0),
    BoundaryCondition('ux', 'dirichlet', 'left_wall',   lambda x,y: 0.0), BoundaryCondition('uy', 'dirichlet', 'left_wall',   lambda x,y: 0.0),
    BoundaryCondition('ux', 'dirichlet', 'right_wall',  lambda x,y: 0.0), BoundaryCondition('uy', 'dirichlet', 'right_wall',  lambda x,y: 0.0),
    BoundaryCondition('ux', 'dirichlet', 'top_lid',     lambda x,y: 1.0), BoundaryCondition('uy', 'dirichlet', 'top_lid',     lambda x,y: 0.0),
    BoundaryCondition('p', 'dirichlet', 'pressure_pin_point',lambda x,y: 0.0)
]
bcs_homog = [BoundaryCondition(bc.field, bc.method, bc.domain_tag, lambda x,y: 0.0) for bc in bcs]
dof_handler.info()

# 2. ============================================================================
#    UFL FORMULATION
# ===============================================================================
rho, dt, theta, mu = Constant(1.0), Constant(0.1), Constant(1.0), Constant(0.01)
velocity_space, pressure_space = FunctionSpace("velocity", ['ux', 'uy']), FunctionSpace("pressure", ['p'])
du, dp = VectorTrialFunction(velocity_space), TrialFunction(pressure_space)
v, q = VectorTestFunction(velocity_space), TestFunction(pressure_space)
u_k, p_k = VectorFunction(name="u_k", field_names=['ux', 'uy'], dof_handler=dof_handler), Function(name="p_k", field_name='p', dof_handler=dof_handler)
u_n, p_n = VectorFunction(name="u_n", field_names=['ux', 'uy'], dof_handler=dof_handler), Function(name="p_n", field_name='p', dof_handler=dof_handler)

# 3. ============================================================================
#    NEW: HIGH-PERFORMANCE SOLVER LOOP
# ===============================================================================
steady_state_tol, max_timesteps = 1e-5, 200
newton_tol, max_newton_iter = 1e-6, 15

u_k.nodal_values.fill(0.0); p_k.nodal_values.fill(0.0)
u_n.nodal_values.fill(0.0); p_n.nodal_values.fill(0.0)
dof_handler.apply_bcs(bcs, u_n, p_n)

# --- NEW: JIT Compile Forms ONCE Before the Loop ---
print("\nJIT compiling Jacobian and Residual forms...")
jacobian_form = (
    rho * dot(du, v) / dt + theta * rho * dot(dot(grad(u_k), du), v) +
    theta * rho * dot(dot(grad(du), u_k), v) + theta * mu * inner(grad(du), grad(v)) -
    dp * div(v) + q * div(du)
) * dx()

residual_form = -(
    rho * dot(u_k - u_n, v) / dt + theta * dot(rho * dot(grad(u_k), u_k), v) +
    (1.0 - theta) * dot(rho * dot(grad(u_n), u_n), v) +
    theta * mu * inner(grad(u_k), grad(v)) +
    (1.0 - theta) * mu * inner(grad(u_n), grad(v)) -
    p_k * div(v) + q * div(u_k)
) * dx()

# The compile step returns high-performance runner objects
jacobian_runner, jacobian_ir = compile_backend(jacobian_form, dof_handler, mixed_element)
residual_runner, residual_ir = compile_backend(residual_form, dof_handler, mixed_element)

# --- NEW: Precompute Geometric Factors ONCE ---
print("Precomputing geometric factors...")
static_kernel_args = dof_handler.precompute_geometric_factors(quad_order=6)

# --- Main Time-Stepping Loop (to reach steady state) ---
for n in range(max_timesteps):
    t = (n + 1) * dt.value
    print(f"\n--- Solving Time Step {n+1} | t = {t:.2f}s ---")

    u_k.nodal_values[:] = u_n.nodal_values[:]
    p_k.nodal_values[:] = p_n.nodal_values[:]
    dof_handler.apply_bcs(bcs, u_k, p_k)

    # --- NEW: Timing the Newton Loop ---
    newton_start_time = time.perf_counter()
    
    for k in range(max_newton_iter):
        # Define the functions with their current data for this iteration
        current_funcs = {'u_k': u_k, 'p_k': p_k, 'u_n': u_n, 'p_n': p_n}
        
        # Execute the pre-compiled runners
        K_loc, _ = jacobian_runner(current_funcs, static_kernel_args)
        _, F_loc = residual_runner(current_funcs, static_kernel_args)
        
        # Assemble the global system from local contributions
        A, R_vec = assemble_system_from_local(K_loc, F_loc, dof_handler, bcs=bcs_homog)

        norm_res = np.linalg.norm(R_vec)
        print(f"  Newton iteration {k+1} | Residual Norm: {norm_res:.3e}")

        if norm_res < newton_tol:
            newton_end_time = time.perf_counter()
            print(f"    Newton converged in {k+1} iterations.")
            print(f"    Newton solver time: {newton_end_time - newton_start_time:.4f} seconds.")
            break
        
        delta_U = sp_la.spsolve(A, R_vec)
        dof_handler.add_to_functions(delta_U, [u_k, p_k])
        dof_handler.apply_bcs(bcs, u_k, p_k)
    else:
        newton_end_time = time.perf_counter()
        print(f"    Newton solver time: {newton_end_time - newton_start_time:.4f} seconds.")
        raise RuntimeError(f"Newton's method did not converge after {max_newton_iter} iterations.")

    # --- Check for steady-state convergence ---
    solution_change = np.linalg.norm(u_k.nodal_values - u_n.nodal_values)
    print(f"  Change in solution (L2 norm): {solution_change:.3e}")
    if solution_change < steady_state_tol and n > 0:
        print(f"\n--- Steady state reached at t={t:.2f}s ---")
        u_n.nodal_values[:] = u_k.nodal_values[:]
        p_n.nodal_values[:] = p_k.nodal_values[:]
        break

    u_n.nodal_values[:], p_n.nodal_values[:] = u_k.nodal_values[:], p_k.nodal_values[:]
else:
    print(f"\n--- Max timesteps ({max_timesteps}) reached. Solution may not be fully steady. ---")

# 4. ============================================================================
#    POST-PROCESSING AND VERIFICATION
# ===============================================================================
print("\nSimulation finished. Generating and saving plots...")

# NEW: Create a directory for the plots if it doesn't exist
output_dir = "cavity_results"
os.makedirs(output_dir, exist_ok=True)

# --- Plot Final Solution Contours and Save them ---
u_n.plot(field='ux', title='U-Velocity (ux) at Steady State (Re=100)', levels=20, cmap='viridis')
plt.savefig(os.path.join(output_dir, "steady_state_velocity_ux.png"), dpi=300)
plt.close()

u_n.plot(field='uy', title='V-Velocity (uy) at Steady State (Re=100)', levels=20, cmap='viridis')
plt.savefig(os.path.join(output_dir, "steady_state_velocity_uy.png"), dpi=300)
plt.close()

p_n.plot(title='Pressure (p) at Steady State (Re=100)', levels=20, cmap='viridis')
plt.savefig(os.path.join(output_dir, "steady_state_pressure.png"), dpi=300)
plt.close()

# --- Generate and Save Quantitative Verification Plot ---
create_verification_plot(dof_handler, u_n, ghia_data_re100)

print(f"\nAll plots saved to the '{output_dir}' directory.")