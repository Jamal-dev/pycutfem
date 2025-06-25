import os
from pathlib import Path
cur_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = "garbage"
filename = Path(cur_dir) / Path("..") / Path("..") / Path("..")/ Path(output_dir) / Path("test_vector_advection_q2.xlsx")
# check if dir exists
os.makedirs(filename.parent, exist_ok=True)
import numpy as np
import pytest

from pycutfem.utils.meshgen      import structured_quad
from pycutfem.core.mesh          import Mesh
from pycutfem.core.dofhandler    import DofHandler
from pycutfem.ufl.functionspace  import FunctionSpace
from pycutfem.ufl.expressions    import VectorTrialFunction, VectorTestFunction
from pycutfem.ufl.expressions    import VectorFunction, Constant, grad, dot
from pycutfem.ufl.measures       import dx
from pycutfem.ufl.forms          import assemble_form
from pycutfem.fem.reference      import get_reference
from pycutfem.fem                import transform
from pycutfem.integration.quadrature import volume           # Gauss points
import logging
import pandas as pd
logging.basicConfig(
    level=logging.INFO,  # show debug messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# --------------------------------------------------------------------------
def build_mesh_Q2():
    nodes, elems, _, corners = structured_quad(1.0, 1.0, nx=1, ny=1, poly_order=2)
    return Mesh(nodes, elems, elements_corner_nodes=corners,
                element_type="quad", poly_order=2)

def reference_matrix_Q2(grad_uk, mesh):
    """Exact element matrix  ∫ (grad uk)_{ij} ϕ_j ϕ_i  dΩ  """
    ref  = get_reference('quad', 2)
    qpts, qwts = volume('quad', 5)               # 3×3 Gauss rule

    n_bs  = 9
    n_comp = 2
    M = np.zeros((n_bs*n_comp, n_bs*n_comp))

    for (xi,eta), w in zip(qpts, qwts):
        N   = ref.shape(xi,eta)                 # (9,)
        J   = transform.jacobian(mesh, 0, (xi,eta))
        det = abs(np.linalg.det(J))
        # block‑diagonal 2×2 structure
        for i in range(2):
            for j in range(2):
                blk = grad_uk[i,j] * np.outer(N, N) * det * w
                r = slice(i*n_bs,(i+1)*n_bs)
                c = slice(j*n_bs,(j+1)*n_bs)
                M[r,c] += blk
    return M

def reference_matrix_Q2_corrected(mesh, dh, uk_nodal_vals):
    """
    Corrected element matrix where grad(uk) is re-calculated at each
    quadrature point.
    """
    ref  = get_reference('quad', 2)
    # Use the same quadrature rule as the assembler for an apples-to-apples comparison
    qpts, qwts = volume('quad', 5)

    n_bs  = 9  # Number of basis functions for Q2 element
    n_comp = 2 # Number of vector components
    M = np.zeros((n_bs * n_comp, n_bs * n_comp))

    # Get the nodal values for the specific element being processed (here, element 0)
    ux_vals = uk_nodal_vals[dh.element_maps['ux'][0]]
    uy_vals = uk_nodal_vals[dh.element_maps['uy'][0]]

    # --- Quadrature Loop ---
    for (xi, eta), w in zip(qpts, qwts):
        # Shape functions and their gradients are evaluated at the current quad point
        N = ref.shape(xi, eta)
        grad_N_ref = ref.grad(xi, eta) # Gradient in reference coords (9, 2)

        # Geometric transformation terms
        J = transform.jacobian(mesh, 0, (xi, eta))
        det_J = abs(np.linalg.det(J))
        J_inv_T = np.linalg.inv(J).T

        # Transform shape function gradient to physical coordinates
        grad_N_phys = grad_N_ref @ J_inv_T # (9, 2)

        # --- THIS IS THE CRITICAL FIX ---
        # Calculate the physical gradient of uk at THIS quadrature point
        grad_ux_at_qpt = grad_N_phys.T @ ux_vals # (2,) -> [d(ux)/dx, d(ux)/dy]
        grad_uy_at_qpt = grad_N_phys.T @ uy_vals # (2,) -> [d(uy)/dx, d(uy)/dy]
        grad_uk_at_qpt = np.vstack([grad_ux_at_qpt, grad_uy_at_qpt]) # (2, 2)
        # --- END OF FIX ---

        # Assemble the 4 blocks of the element matrix using the local gradient
        for i in range(2):
            for j in range(2):
                # The term grad_uk[i,j] is now specific to this quad point
                blk = grad_uk_at_qpt[i, j] * np.outer(N, N) * det_J * w
                r = slice(i * n_bs, (i + 1) * n_bs)
                c = slice(j * n_bs, (j + 1) * n_bs)
                M[r, c] += blk
    return M

def test_dot_dot_grad_q2():
    # ---------- set‑up --------------------------------------------------
    mesh = build_mesh_Q2()
    dh   = DofHandler({'ux': mesh, 'uy': mesh}, method='cg')

    V    = FunctionSpace("V", ['ux', 'uy'])
    du   = VectorTrialFunction(V,dof_handler=dh)
    v    = VectorTestFunction(V,dof_handler=dh)

    np.random.seed(321)
    uk   = VectorFunction("u_k", ['ux','uy'], dh)
    uk.nodal_values[:] = np.random.rand(*uk.nodal_values.shape)

    # ---------- assemble with current compiler -------------------------
    lhs = dot(dot(grad(uk), du), v) * dx()

    K_pc, _ = assemble_form(lhs == Constant(0.0)*v[0]*dx(), dh, quad_degree=5)
    K_pc = K_pc.toarray()

    # ---------- analytical reference -----------------------------------
    ref = get_reference('quad', 2)
    # grad uk is constant in our one‑element mesh -----------------------
    xi_eta = (0.0,0.0)
    G = ref.grad(*xi_eta) @ np.linalg.inv(transform.jacobian(mesh,0,xi_eta)).T  # (9,2)

    grad_ux = G.T @ uk.nodal_values[dh.element_maps['ux'][0]]
    grad_uy = G.T @ uk.nodal_values[dh.element_maps['uy'][0]]
    grad_uk = np.vstack([grad_ux, grad_uy])       # (2,2)

    K_ref = reference_matrix_Q2(grad_uk, mesh)
    K_ref = reference_matrix_Q2_corrected(mesh, dh, uk.nodal_values)

    # ---------- comparison ---------------------------------------------
    
    with pd.ExcelWriter(filename) as writer:
            pd.DataFrame(K_pc).to_excel(writer, sheet_name='pycutfem', index=False, header=False)
            pd.DataFrame(K_ref).to_excel(writer, sheet_name='whiteboxTest', index=False, header=False)
    np.testing.assert_allclose(K_pc, K_ref, rtol=1e-12, atol=1e-12)
