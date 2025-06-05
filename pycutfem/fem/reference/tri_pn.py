"""
pycutfem.fem.reference.tri_pn
Arbitrary order barycentric Lagrange basis on reference triangle.
"""
from functools import lru_cache
import sympy as sp
import numpy as np

# --- Pn Triangular Basis Functions (Bernstein Basis) ---
# @lru_cache(maxsize=None)
# def tri_pn(n: int):
#     """
#     Return lambdified (shape, grad_components, hess_components, laplacian)
#     callables for Pn (triangular Bernstein-Bezier) elements of order n.

#     grad_components: Tuple (dPhi_dxi_lambda, dPhi_deta_lambda)
#     hess_components: Tuple (d2Phi_dxixi_lambda, d2Phi_detaeta_lambda, d2Phi_dxieta_lambda)
#     Each lambda function returns a vector of values for all basis functions.
#     """
#     xi_sym, eta_sym = sp.symbols("xi eta")
#     L1_sym = 1 - xi_sym - eta_sym
#     L2_sym = xi_sym
#     L3_sym = eta_sym
    
#     basis_sym_list = []
#     # For gradients (component-wise for each basis function)
#     dPhi_dxi_sym_list = []
#     dPhi_deta_sym_list = []
#     # For Hessian components (component-wise for each basis function)
#     d2Phi_dxixi_sym_list = []
#     d2Phi_detaeta_sym_list = []
#     d2Phi_dxieta_sym_list = []
#     # For Laplacians (for each basis function)
#     laplacian_Phi_sym_list = []

#     # Loop order matches the one from the previous corrected version for consistent node mapping
#     for j_L3 in range(n + 1):  # Power of L3_sym (eta-like)
#         for i_L2 in range(n + 1 - j_L3):  # Power of L2_sym (xi-like)
#             k_L1 = n - i_L2 - j_L3  # Power of L1_sym (1-xi-eta-like)
            
#             # Bernstein polynomial coefficient
#             coeff = sp.binomial(n, i_L2) * sp.binomial(n - i_L2, j_L3) # n! / (i! j! k!)
            
#             phi_sym = coeff * (L1_sym**k_L1) * (L2_sym**i_L2) * (L3_sym**j_L3)
#             basis_sym_list.append(phi_sym)

#             # Gradients
#             dphi_dxi = sp.simplify(sp.diff(phi_sym, xi_sym))
#             dphi_deta = sp.simplify(sp.diff(phi_sym, eta_sym))
#             dPhi_dxi_sym_list.append(dphi_dxi)
#             dPhi_deta_sym_list.append(dphi_deta)

#             # Hessian components
#             ddphi_dxixi = sp.simplify(sp.diff(phi_sym, xi_sym, 2))
#             ddphi_detaeta = sp.simplify(sp.diff(phi_sym, eta_sym, 2))
#             ddphi_dxieta = sp.simplify(sp.diff(phi_sym, xi_sym, eta_sym))
#             d2Phi_dxixi_sym_list.append(ddphi_dxixi)
#             d2Phi_detaeta_sym_list.append(ddphi_detaeta)
#             d2Phi_dxieta_sym_list.append(ddphi_dxieta)

#             # Laplacian (w.r.t. reference coordinates xi, eta)
#             laplacian_phi = sp.simplify(ddphi_dxixi + ddphi_detaeta)
#             laplacian_Phi_sym_list.append(laplacian_phi)

#     # Lambdify
#     shape_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(basis_sym_list), "numpy")
    
#     dPhi_dxi_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(dPhi_dxi_sym_list), "numpy")
#     dPhi_deta_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(dPhi_deta_sym_list), "numpy")

#     d2Phi_dxixi_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(d2Phi_dxixi_sym_list), "numpy")
#     d2Phi_detaeta_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(d2Phi_detaeta_sym_list), "numpy")
#     d2Phi_dxieta_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(d2Phi_dxieta_sym_list), "numpy")
    
#     laplacian_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(laplacian_Phi_sym_list), "numpy")

#     return (shape_lambda,
#             (dPhi_dxi_lambda, dPhi_deta_lambda), 
#             (d2Phi_dxixi_lambda, d2Phi_detaeta_lambda, d2Phi_dxieta_lambda), 
#             laplacian_lambda)




# quad_qn and _lagrange_basis_1d would remain the same as they already produce Lagrange basis.

@lru_cache(maxsize=None)
def tri_pn(n: int): # n is the polynomial order k for Pk elements
    if n < 0:
        raise ValueError("Polynomial order n must be non-negative.")
    xi_sym, eta_sym = sp.symbols("xi eta")
    
    # 1. Define Pk nodal points on the reference triangle (0,0)-(1,0)-(0,1)
    #    Order: iterate eta-like rows, then xi-like within rows.
    nodes_ref_coords = []
    if n == 0: # P0 element has one node
        nodes_ref_coords.append((sp.S(0), sp.S(0))) # Arbitrary point for P0, basis is constant 1.
                                                   # Could also be centroid (1/3, 1/3).
    else:
        for j_level in range(n + 1):  # Corresponds to eta_node = j_level / n
            for i_level in range(n + 1 - j_level):  # Corresponds to xi_node = i_level / n
                nodes_ref_coords.append((sp.Rational(i_level, n), sp.Rational(j_level, n)))
    
    num_nodes = len(nodes_ref_coords)
    expected_num_nodes = (n + 1) * (n + 2) // 2 if n > 0 else 1
    if num_nodes != expected_num_nodes: # Should not be hit if P0 logic is correct
        raise RuntimeError(f"Internal error: Mismatch in Pk node count for order n={n}. "
                           f"Generated {num_nodes}, expected {expected_num_nodes}")

    # 2. Define a monomial basis for polynomials of degree n in 2D
    #    Order: 1, xi, eta, xi^2, xi*eta, eta^2, ...
    monomials_sym = []
    if n == 0: # P0 monomial basis is just 1
        monomials_sym.append(sp.S(1))
    else:
        for total_degree in range(n + 1):
            for pow_xi in range(total_degree + 1):
                pow_eta = total_degree - pow_xi
                monomials_sym.append(xi_sym**pow_xi * eta_sym**pow_eta)

    if len(monomials_sym) != num_nodes : 
        raise RuntimeError(
            f"Internal error: Mismatch between number of nodes ({num_nodes}) "
            f"and number of monomials ({len(monomials_sym)}) for order n={n}."
        )

    # 3. Construct the Vandermonde-like matrix V
    #    V[i_node, j_monomial] = monomial_j(node_i_xi, node_i_eta)
    V_matrix = sp.zeros(num_nodes, num_nodes)
    for i_node, (node_xi, node_eta) in enumerate(nodes_ref_coords):
        for j_monomial, monomial in enumerate(monomials_sym):
            V_matrix[i_node, j_monomial] = monomial.subs({xi_sym: node_xi, eta_sym: node_eta})

    # 4. Invert V.T to get coefficients for Lagrange basis functions.
    #    If phi_k = sum_m C[k,m]*M_m, then C * V.T = I, so C = (V.T)^-1
    #    The k-th row of C gives coefficients for the k-th Lagrange polynomial.
    coeffs_matrix = sp.Matrix([]) 
    if num_nodes > 0 :
        try:
            # Correct way to get coefficients for Lagrange basis
            coeffs_matrix = (V_matrix.T).inv() 
        except sp.matrices.common.NonInvertibleMatrixError: # More specific exception
             raise RuntimeError(f"Vandermonde.T matrix is singular for tri_pn order n={n}. "
                                 "Check node definitions or monomial basis.")
        except Exception as e:
             raise RuntimeError(f"Matrix inversion (V.T).inv() failed for tri_pn order n={n}. Original Error: {e}")
    
    # 5. Construct the symbolic Lagrange basis functions
    basis_sym_list = []
    monomials_matrix_col = sp.Matrix(monomials_sym) # Column vector of monomials
    
    if n == 0 : # P0 basis is just 1
        basis_sym_list.append(sp.S(1))
    else:
        for k_node_idx in range(num_nodes):
            # k_node_idx-th Lagrange basis function
            coeffs_for_phi_k_row_vec = coeffs_matrix.row(k_node_idx) 
            phi_k_sym = sp.simplify((coeffs_for_phi_k_row_vec * monomials_matrix_col)[0,0]) # Dot product
            basis_sym_list.append(phi_k_sym)
    
    # --- Calculate derivatives ---
    dPhi_dxi_sym_list, dPhi_deta_sym_list = [], []
    d2Phi_dxixi_sym_list, d2Phi_detaeta_sym_list, d2Phi_dxieta_sym_list = [], [], []
    laplacian_Phi_sym_list = []

    for phi_sym in basis_sym_list:
        dphi_dxi = sp.simplify(sp.diff(phi_sym, xi_sym))
        dphi_deta = sp.simplify(sp.diff(phi_sym, eta_sym))
        dPhi_dxi_sym_list.append(dphi_dxi); dPhi_deta_sym_list.append(dphi_deta)

        ddphi_dxixi = sp.simplify(sp.diff(phi_sym, xi_sym, 2))
        ddphi_detaeta = sp.simplify(sp.diff(phi_sym, eta_sym, 2))
        ddphi_dxieta = sp.simplify(sp.diff(phi_sym, xi_sym, eta_sym))
        d2Phi_dxixi_sym_list.append(ddphi_dxixi)
        d2Phi_detaeta_sym_list.append(ddphi_detaeta)
        d2Phi_dxieta_sym_list.append(ddphi_dxieta)

        laplacian_phi = sp.simplify(ddphi_dxixi + ddphi_detaeta)
        laplacian_Phi_sym_list.append(laplacian_phi)

    # Lambdify
    shape_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(basis_sym_list), "numpy")
    dPhi_dxi_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(dPhi_dxi_sym_list), "numpy")
    dPhi_deta_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(dPhi_deta_sym_list), "numpy")
    d2Phi_dxixi_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(d2Phi_dxixi_sym_list), "numpy")
    d2Phi_detaeta_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(d2Phi_detaeta_sym_list), "numpy")
    d2Phi_dxieta_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(d2Phi_dxieta_sym_list), "numpy")
    laplacian_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(laplacian_Phi_sym_list), "numpy")
    
    return (shape_lambda, 
            (dPhi_dxi_lambda, dPhi_deta_lambda), 
            (d2Phi_dxixi_lambda, d2Phi_detaeta_lambda, d2Phi_dxieta_lambda), 
            laplacian_lambda)


if __name__ == "__main__":
    # Example usage
    n = 2  # Order of the polynomial
    shape, grad_components, hess_components, laplacian = tri_pn(n)
    
    xi, eta = 0.25, 0.25  # Example point in reference triangle
    print("Shape functions:", shape(xi, eta))
    print("Gradients:", grad_components[0](xi, eta), grad_components[1](xi, eta))
    print("Hessian components:", hess_components[0](xi, eta), hess_components[1](xi, eta), hess_components[2](xi, eta))
    print("Laplacian:", laplacian(xi, eta))