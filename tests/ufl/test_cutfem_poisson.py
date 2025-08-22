from matplotlib.pylab import norm
from pycutfem.utils import boundary
from pyparsing import C
import pytest
import numpy as np
import scipy.sparse.linalg as spla

# --- Core imports ---
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.meshgen import structured_quad
from pycutfem.utils.bitset import BitSet
from pycutfem.utils.domain_manager import get_domain_bitset
from pycutfem.io.vtk import export_vtk

# --- UFL-like imports ---
from pycutfem.ufl.expressions import (TrialFunction, TestFunction, grad, inner, jump,dot,
                             ElementWiseConstant, 
                             Jump,
                             Constant, FacetNormal, CellDiameter, Derivative, Pos, Neg,
                             Function, VectorFunction)
from pycutfem.ufl.measures import dx, ds, dInterface, dGhost
from pycutfem.ufl.forms import BoundaryCondition, assemble_form, Equation

# --- Level Set and Cut Ratio imports ---
from pycutfem.core.levelset import CircleLevelSet
from pycutfem.core.geometry import hansbo_cut_ratio
from pycutfem.io.visualization import plot_mesh_2
from pycutfem.fem.mixedelement import MixedElement
from scipy.sparse.linalg import eigsh,  ArpackNoConvergence
from scipy.sparse import csr_matrix, bmat
import os


def smallest_eigs_safe(K, k=3):
    try:
        return eigsh(K, k=k, which="SA", return_eigenvectors=False, tol=1e-10, maxiter=10000)
    except ArpackNoConvergence:
        return np.linalg.eigvalsh(K.toarray())[:k]

def _dn(expr,n):
    """Normal derivative  n·∇expr  on an (interior) edge."""
    return n[0] * Derivative(expr, 1, 0) + n[1] * Derivative(expr, 0, 1)
    # return dot(grad(expr), n)

def grad_inner(u, v, n):
    """⟨∂ₙu, ∂ₙv⟩  (scalar or 2‑D vector)."""
    if getattr(u, "num_components", 1) == 1:      # scalar
        return _dn(u, n) * _dn(v, n)

    if u.num_components == v.num_components == 2: # vector
        return _dn(u[0], n) * _dn(v[0], n) + _dn(u[1], n) * _dn(v[1], n)


@pytest.mark.parametrize("backend", ["jit", "python"])
def test_cutfem_poisson_interface(backend):
    """
    Tests the solver for a Poisson interface problem using CutFEM.
    -α∇²u = f
    This version uses the final, robust, BitSet-driven framework.
    """
    # 1. Setup Mesh and Level Set
    poly_order = 1
    L,H = 3.0, 3.0
    nx, ny = 30, 30
    ghost_parameter = 0.5
    beta =20  # Stabilization parameter
    gamma_G = Constant(ghost_parameter)
    nodes, elems, _, corners = structured_quad(L,H, nx=nx, ny=ny, poly_order=poly_order, offset=[-1.5,-1.5])
    mesh = Mesh(nodes, element_connectivity=elems, elements_corner_nodes=corners, poly_order=poly_order, element_type='quad')
    c_x, c_y = 0.0, 0.0
    radius = 1.0
    level_set = CircleLevelSet(center=(c_x, c_y), radius=radius)

    # 2. Apply general boundary tags FIRST
    boundary_tags = {
        'left': lambda x, y: np.isclose(x, -L/2.0),
        'right': lambda x, y: np.isclose(x, L/2.0),
        'bottom': lambda x, y: np.isclose(y, -H/2.0),
        'top': lambda x, y: np.isclose(y, H/2.0),
    }
    # Classify mesh elements and edges against the level set
    mesh.classify_elements(level_set)
    mesh.classify_edges(level_set)
    mesh.build_interface_segments(level_set)
    mesh.tag_boundary_edges(boundary_tags)
    # plot_mesh_2(mesh, level_set=level_set)

    # plot_mesh_2(mesh, level_set=level_set)

    # 2. Create BitSets for the integration domains
    inside_elements  = mesh.element_bitset("inside")
    outside_elements  = mesh.element_bitset("outside")
    cut_elements  = mesh.element_bitset("cut")
    has_outside_elements = outside_elements | cut_elements
    has_inside_elements = inside_elements | cut_elements
    cut_domain = mesh.element_bitset("cut")
    # Ghost penalty is applied on faces between cut cells and interior cells
    ghost_domain = mesh.edge_bitset("ghost_neg") | mesh.edge_bitset("ghost_both") 
    print(f"Negative elements: {inside_elements}, Positive elements: {outside_elements}, Cut elements: {cut_elements}")
    print(f"Has positive elements: {has_outside_elements}, Has negative elements: {has_inside_elements}")
    for tag, bitset in mesh._edge_bitsets.items():
        print(f"Boundary tag '{tag}': {bitset.cardinality()}")
    # plot_mesh_2(mesh,level_set=level_set)
    


    # 3. Define the DofHandler for the two-field problem
    # The field names are now just labels; their meaning is defined by their use in the weak form.
    me = MixedElement(mesh, field_specs={'u_outside' : poly_order, 'u_inside': poly_order})
    # fe_map = {'u_outside': mesh, 'u_inside': mesh}
    dof_handler = DofHandler(me, method='cg')

    dof_handler.tag_dofs_from_element_bitset("inactive_inside", "u_outside", "inside", strict=True)
    dof_handler.tag_dofs_from_element_bitset("inactive_outside", "u_inside", "outside", strict=True)


        # --- element sets (unchanged) ---
    

    # --- ghost edge sets: split by side ---
    g_pos  = mesh.edge_bitset("ghost_pos")
    g_neg  = mesh.edge_bitset("ghost_neg")
    g_both = mesh.edge_bitset("ghost_both")
    g_interface = mesh.edge_bitset("interface")
    ghost_pos = g_pos | g_both | g_interface
    ghost_neg = g_neg | g_both | g_interface

    # --- measures ---
    dx_pos  = dx(defined_on=has_outside_elements, level_set=level_set, metadata={'side': '+', 'q': poly_order+2})
    dx_neg  = dx(defined_on=has_inside_elements, level_set=level_set, metadata={'side': '-', 'q': poly_order+2})
    dGamma  = dInterface(defined_on=cut_elements, level_set=level_set, metadata={'q': poly_order+2})
    dGhost_pos = dGhost(defined_on=ghost_pos, level_set=level_set, metadata={'q': poly_order+2, "derivs": {(0,1),(1,0)}})
    dGhost_neg = dGhost(defined_on=ghost_neg, level_set=level_set, metadata={'q': poly_order+2, "derivs": {(0,1),(1,0)}})

    # --- trial/test ---
    u_pos, v_pos = TrialFunction('u_outside', dof_handler, name='u_pos_trial'), TestFunction('u_outside', dof_handler, name='u_pos_test')
    u_neg, v_neg = TrialFunction('u_inside',  dof_handler, name='u_neg_trial'), TestFunction('u_inside',  dof_handler, name='u_neg_test')

    # --- coefficients / constants (as you had) ---
    alpha_vals = np.zeros(len(mesh.elements_list))
    alpha_vals[outside_elements.to_indices()] = 1.0
    alpha_vals[inside_elements.to_indices()] = 20.0
    alpha_vals[cut_elements.to_indices()] = 1.0  # convention
    alpha = ElementWiseConstant(alpha_vals)

    h    = CellDiameter()
    stab = Constant(20.0 * (20.0 + 1.0)) / h  # you can keep this

    # OPTIONAL: Hansbo scaling for the interface penalty (helps slivers)
    from pycutfem.core.geometry import hansbo_cut_ratio
    theta_min = 1.0e-3
    alpha_hansbo = 0.5
    theta_plus  = np.clip(hansbo_cut_ratio(mesh, level_set, side='+'), theta_min, 1.0)
    theta_minus = np.clip(hansbo_cut_ratio(mesh, level_set, side='-'), theta_min, 1.0)
    beta_scale = 0.5*(theta_plus**(-alpha_hansbo) + theta_minus**(-alpha_hansbo))
    stab = Constant(20.0 * (20.0 + 1.0)) * ElementWiseConstant(beta_scale) / h

    # --- average fluxes (as you had) ---
    n = FacetNormal()
    alpha_pos = Pos(alpha)
    alpha_neg = Neg(alpha)
    avg_flux_u = -0.5 * ( alpha_pos * dot(grad(u_pos), n) - alpha_neg * dot(grad(u_neg), n) )
    avg_flux_v = -0.5 * ( alpha_pos * dot(grad(v_pos), n) - alpha_neg * dot(grad(v_neg), n) )

    # --- core bilinear ---
    a  = inner(alpha_pos * grad(u_pos), grad(v_pos)) * dx_pos
    a += inner(alpha_neg * grad(u_neg), grad(v_neg)) * dx_neg

    # --- CORRECT ghost stabilization: interior jump on each side separately ---
    #     i.e., [∇u_pos]·[∇v_pos] on ghost faces of the + side, and similarly for - side
    a += gamma_G * h * inner(Jump(grad(u_pos), grad(u_neg)), Jump(grad(v_pos), grad(v_neg))) * dGhost_pos
    a += gamma_G * h * inner(Jump(grad(u_pos), grad(u_neg)), Jump(grad(v_pos), grad(v_neg))) * dGhost_neg

    # --- Interface Nitsche terms (unchanged algebra) ---
    jump_u = Jump(u_pos, u_neg)
    jump_v = Jump(v_pos, v_neg)
    a += ( avg_flux_u * jump_v + avg_flux_v * jump_u + stab * jump_u * jump_v ) * dGamma

    # --- RHS
    # to match the NGSolve notebook: f = [1, 0]
    f  = Constant(1.0) * v_neg * dx_neg
    # f += Constant(1.0) * v_pos * dx_pos    # <- keep it commented if you want the exact NGSolve case

    equation = Equation(a, f)


    # 7. Define Boundary Conditions
    bcs = [
        BoundaryCondition('u_outside', 'dirichlet', 'inactive_inside', lambda x, y: 0.0),
        BoundaryCondition('u_inside', 'dirichlet' , 'inactive_outside', lambda x, y: 0.0),
        BoundaryCondition('u_outside', 'dirichlet', 'left', lambda x, y: 0.0),
        BoundaryCondition('u_outside', 'dirichlet', 'right', lambda x, y: 0.0),
        BoundaryCondition('u_outside', 'dirichlet', 'bottom', lambda x, y: 0.0),
        BoundaryCondition('u_outside', 'dirichlet', 'top', lambda x, y: 0.0),
        BoundaryCondition('u_inside', 'dirichlet', 'left', lambda x, y: 0.0),
        BoundaryCondition('u_inside', 'dirichlet', 'right', lambda x, y: 0.0),
        BoundaryCondition('u_inside', 'dirichlet', 'bottom', lambda x, y: 0.0),
        BoundaryCondition('u_inside', 'dirichlet', 'top', lambda x, y: 0.0),    
    ]
    
    # 8. Assemble and Solve
    system = equation
    K, F = assemble_form(system, dof_handler=dof_handler, bcs=bcs, quad_order=poly_order + 2, backend=backend)
    # m_form = Constant(1.0) * v_pos * dx_pos + Constant(1.0) * v_neg * dx_neg          # linear form
    # _,m_vec  = assemble_form(Equation(None, m_form),     # returns vector with same dof layout as K,F
    #                     dof_handler=dof_handler, bcs=[], backend=backend)
    # print(f"m_vec.shape: {m_vec.shape}")
    # m = np.asarray(m_vec).reshape(-1, 1)
    # zero11 = csr_matrix((1,1))
    # K_aug  = bmat([[K,     csr_matrix(m)],
    #             [csr_matrix(m.T), zero11]], format='csr')
    # F_aug  = np.concatenate([F, [0.0]])

    # u_aug = spla.spsolve(K_aug, F_aug)
    # u     = u_aug[:-1]    # solution
    # lam   = u_aug[-1]     # the multiplier (can ignore)

    # mean_val = float(m.T @ u)
    # print(f"mean(u+) + mean(u-) = {mean_val}, should be ≈ 0")   # ≈ 0
    # solution_vec = u.reshape(-1)


    Kd = K.astype(float).tocsr()
    # # 1) Which rows are (near) empty after BC elimination?
    # row_norms = np.linalg.norm(Kd.toarray(), axis=1)
    # suspect = int(np.argmin(row_norms))
    # print("Min row-norm DOF:", suspect, "norm:", row_norms[suspect])

    # # 2) Smallest eigenvector shows the “floating” direction
    # w, v = eigsh(Kd, k=1, which="SM")
    # x = v[:, 0]
    # i_max = int(np.argmax(np.abs(x)))
    # print("Largest component in smallest-eigvec → DOF:", i_max, "ampl:", x[i_max])

    # # 3) Map to (field, node, coords), and show whether *you* tagged it
    # field, node = dof_handler._dof_to_node_map[i_max]
    # xy = dof_handler.fe_map[field].nodes_x_y_pos[node]
    # print(f"Suspect DOF belongs to field='{field}', node={node}, coords={tuple(xy)}")
    # print("Is tagged inactive_inside?:", i_max in dof_handler.dof_tags.get("inactive_inside", set()))
    # print("Is tagged inactive_outside?:", i_max in dof_handler.dof_tags.get("inactive_outside", set()))

    # ones = np.ones(K.shape[0])
    # r = np.linalg.norm((K @ ones))
    # print(f"||K·1|| = {r}, should be ~ 1e-12–1e-14")  # should be ~ 1e-12–1e-14

    # w, v = eigsh(K, k=1, which="SM")
    # corr = abs(v[:,0] @ ones) / (np.linalg.norm(v[:,0]) * np.linalg.norm(ones))
    # print(f"corr(smallest‐eigvec, 1) ≈ {corr}, should be ~ 1.0")  # ~ 1.0

    # # Smallest few eigenvalues of symmetric K (shift-invert is not needed here)
    # w = smallest_eigs_safe(Kd, k=3)
    # lam_min, lam_max = float(w[0]), float(w[-1])
    # tol = 200 * np.finfo(float).eps * max(1.0, lam_max)  # relative PSD tol
    # print(f"Smallest eigenvalue: {lam_min}, Largest eigenvalue: {lam_max}, Tolerance: {tol}")
    # assert lam_min >= -tol, f"Smallest eigenvalue {lam_min} is below tolerance {tol}!"

   

    free = np.where(K.getnnz(axis=1) == 0)[0]
    print("Zero-row DOFs:", free)


    # Check the rank to ensure the system is solvable
    matrix_rank = np.linalg.matrix_rank(K.toarray())
    print(f"\nMatrix Info: Shape={K.shape}, Rank={matrix_rank}")
    assert matrix_rank == K.shape[0], f"Matrix is rank-deficient! Expected full rank {K.shape[0]}, got {matrix_rank}."


    try:
        solution_vec = spla.spsolve(K.tocsc(), F)
    except RuntimeError as e:
        print(f"Solver failed: {e}. The matrix might be singular.")
        # As a fallback, use a least-squares solver
        solution_vec = spla.lsqr(K, F)[0]


    # (A) Make field Functions
    u_out = Function(name="u_outside", field_name="u_outside", dof_handler=dof_handler)
    u_in  = Function(name="u_inside",  field_name="u_inside",  dof_handler=dof_handler)

    # (B) Robust scatter from global → field-local for each Function
    def scatter_global(func, u_global, dof_handler):
        gdofs = np.asarray(dof_handler.get_field_slice(func.field_name), dtype=int)
        func.set_nodal_values(gdofs, u_global[gdofs])


    scatter_global(u_out, solution_vec, dof_handler)
    scatter_global(u_in,  solution_vec, dof_handler)
    u_in.plot(title="u_inside", mask = inside_elements)
    phi_vals = level_set.evaluate_on_nodes(mesh)

    output_dir = "two_alpha_results"
    os.makedirs(output_dir, exist_ok=True)
    vtk_path = os.path.join(output_dir, f"ewc_solution_cycle.vtu")
    print(f"Writing solution to {vtk_path}")

    export_vtk(
        vtk_path,
        mesh=mesh,
        dof_handler=dof_handler,
        functions={
            "u_outside": u_out,
            "u_inside":  u_in,
            "phi":       phi_vals,   # for masking in ParaView
        },
    )


    # 9. Verification
    assert np.linalg.norm(solution_vec) > 1e-8, "Solution is trivial (zero)."
    
    center_node_id = -1
    for i, node_data in enumerate(mesh.nodes_list):
        if np.isclose(node_data.x, c_x) and np.isclose(node_data.y, c_y):
            center_node_id = i
            break
            
    assert center_node_id != -1, "Could not find the center node of the mesh."

    u_pos_dof = dof_handler.dof_map['u_inside'][center_node_id]
    center_val = solution_vec[u_pos_dof]
    print(f"Solution at center (inside domain): {center_val:.4f}")
    assert center_val > 0, "Expected a positive solution at the center."