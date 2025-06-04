"""
pycutfem.fem.reference.tri_pn
Arbitrary order barycentric Lagrange basis on reference triangle.
"""
from functools import lru_cache
import sympy as sp
import numpy as np

@lru_cache(maxsize=None)
def tri_pn(n: int): # n is the polynomial order k for Pk elements
    xi, eta = sp.symbols("xi eta")
    
    # Define barycentric coordinates for the reference triangle with vertices
    # V0 at (xi=0, eta=0) -> L1 is 1
    # V1 at (xi=1, eta=0) -> L2 is 1
    # V2 at (xi=0, eta=1) -> L3 is 1
    L1_sym = 1 - xi - eta  # Corresponds to vertex V0 (0,0)
    L2_sym = xi            # Corresponds to vertex V1 (1,0)
    L3_sym = eta            # Corresponds to vertex V2 (0,1)
    
    basis_expressions = []
    
    # Loop order to match a common nodal enumeration:
    # Outer loop for j_power_L3 (eta-like, controlling "rows" from bottom up)
    # Inner loop for i_power_L2 (xi-like, controlling nodes within a "row")
    for j_L3 in range(n + 1):  # j_L3 is the power of L3_sym (eta)
        for i_L2 in range(n + 1 - j_L3):  # i_L2 is the power of L2_sym (xi)
            k_L1 = n - i_L2 - j_L3  # k_L1 is the power of L1_sym (1-xi-eta)
            
            # Bernstein basis polynomial B_ijk = n!/(i!j!k!) * L2^i * L3^j * L1^k
            # The coefficient n!/(i!j!k!) is sp.binomial(n, k_L1) * sp.binomial(n - k_L1, i_L2)
            # or equivalently sp.binomial(n, i_L2) * sp.binomial(n - i_L2, j_L3)
            
            coeff = sp.binomial(n, i_L2) * sp.binomial(n - i_L2, j_L3)
            
            term = coeff * (L1_sym**k_L1) * (L2_sym**i_L2) * (L3_sym**j_L3)
            basis_expressions.append(term)

    # Sanity check the number of basis functions generated
    expected_num_basis_funcs = (n + 1) * (n + 2) // 2
    if len(basis_expressions) != expected_num_basis_funcs:
        # This should not happen if the loop bounds are correct
        raise RuntimeError(
            f"Internal error in tri_pn: generated {len(basis_expressions)} basis functions "
            f"for order n={n}, but expected {expected_num_basis_funcs}."
        )

    # Calculate gradients for each basis function
    dbasis_matrices_sym = []
    for phi_expr in basis_expressions:
        # Gradient with respect to (xi, eta)
        grad_phi_sym = sp.Matrix([sp.diff(phi_expr, xi), sp.diff(phi_expr, eta)]) # This is a 2x1 column matrix
        dbasis_matrices_sym.append(grad_phi_sym)
            
    # Lambdify the symbolic expressions
    # sp.Matrix(basis_expressions) will be a (num_basis_funcs x 1) column matrix of shape functions
    shape_lambda = sp.lambdify((xi, eta), sp.Matrix(basis_expressions), "numpy")
    
    # sp.Matrix.hstack(*dbasis_matrices_sym) creates a matrix where columns are the gradient vectors.
    # Resulting matrix shape: (2, num_basis_functions)
    # Example: Column 0 is [d(phi_0)/dxi, d(phi_0)/deta]^T
    grad_matrix_lambda = sp.lambdify((xi, eta), sp.Matrix.hstack(*dbasis_matrices_sym), "numpy")
    
    return shape_lambda, grad_matrix_lambda
