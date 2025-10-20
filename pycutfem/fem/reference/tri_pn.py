from functools import lru_cache
import sympy as sp
import numpy as np

@lru_cache(maxsize=None)
def tri_pn(n: int, max_deriv_order: int = 2):
    """
    Return lambdified shape functions and derivatives up to max_deriv_order for Pn triangular elements.

    Args:
        n: Polynomial order of the Pn element.
        max_deriv_order: Maximum total derivative order to compute (default 2).

    Returns:
        tuple: (shape_lambda, deriv_lambdas)
            - shape_lambda: Callable giving shape function values [phi_1, ..., phi_N] at (xi, eta).
            - deriv_lambdas: Dict with keys (alpha_xi, alpha_eta), values are callables giving
                             derivative values [D^alpha phi_1, ..., D^alpha phi_N] at (xi, eta).
    """
    if n < 0:
        raise ValueError("Polynomial order n must be non-negative.")
    xi_sym, eta_sym = sp.symbols("xi eta")

    # 1. Define Pn nodal points on the reference triangle (0,0)-(1,0)-(0,1)
    nodes_ref_coords = []
    if n == 0:  # P0 element has one node
        nodes_ref_coords.append((sp.S(0), sp.S(0)))  # Could use centroid (1/3, 1/3)
    else:
        for j_level in range(n + 1):  # eta-like rows
            for i_level in range(n + 1 - j_level):  # xi-like within rows
                nodes_ref_coords.append((sp.Rational(i_level, n), sp.Rational(j_level, n)))

    num_nodes = len(nodes_ref_coords)
    expected_num_nodes = (n + 1) * (n + 2) // 2 if n > 0 else 1
    if num_nodes != expected_num_nodes:
        raise RuntimeError(f"Internal error: Mismatch in Pn node count for order n={n}. "
                           f"Generated {num_nodes}, expected {expected_num_nodes}")

    # 2. Define monomial basis for polynomials of degree <= n in 2D
    monomials_sym = []
    if n == 0:
        monomials_sym.append(sp.S(1))
    else:
        for total_degree in range(n + 1):
            for pow_xi in range(total_degree + 1):
                pow_eta = total_degree - pow_xi
                monomials_sym.append(xi_sym**pow_xi * eta_sym**pow_eta)

    if len(monomials_sym) != num_nodes:
        raise RuntimeError(f"Internal error: Mismatch between number of nodes ({num_nodes}) "
                           f"and number of monomials ({len(monomials_sym)}) for order n={n}.")

    # 3. Construct the Vandermonde-like matrix V
    V_matrix = sp.zeros(num_nodes, num_nodes)
    for i_node, (node_xi, node_eta) in enumerate(nodes_ref_coords):
        for j_monomial, monomial in enumerate(monomials_sym):
            V_matrix[i_node, j_monomial] = monomial.subs({xi_sym: node_xi, eta_sym: node_eta})

    # 4. Compute coefficients for Lagrange basis functions
    try:
        coeffs_matrix = (V_matrix.T).inv()
    except sp.matrices.common.NonInvertibleMatrixError:
        raise RuntimeError(f"Vandermonde.T matrix is singular for tri_pn order n={n}.")
    except Exception as e:
        raise RuntimeError(f"Matrix inversion failed for tri_pn order n={n}. Error: {e}")

    # 5. Construct symbolic Lagrange basis functions
    basis_sym_list = []
    monomials_matrix_col = sp.Matrix(monomials_sym)
    if n == 0:
        basis_sym_list.append(sp.S(1))
    else:
        for k_node_idx in range(num_nodes):
            coeffs_for_phi_k_row_vec = coeffs_matrix.row(k_node_idx)
            phi_k_sym = sp.simplify((coeffs_for_phi_k_row_vec * monomials_matrix_col)[0, 0])
            basis_sym_list.append(phi_k_sym)

    # 6. Generate multi-indices for derivatives up to max_deriv_order
    multi_indices = [(i, j) for i in range(max_deriv_order + 1)
                    for j in range(max_deriv_order + 1) if i + j <= max_deriv_order]

    # 7. Compute symbolic derivatives
    derivs_2d_sym_dict = {}
    for alpha in multi_indices:
        alpha_xi, alpha_eta = alpha
        derivs_alpha = [sp.simplify(sp.diff(phi_sym, xi_sym, alpha_xi, eta_sym, alpha_eta))
                        for phi_sym in basis_sym_list]
        derivs_2d_sym_dict[alpha] = derivs_alpha

    # 8. Lambdify shape functions and derivatives
    shape_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(basis_sym_list), "numpy")
    deriv_lambdas = {alpha: sp.lambdify((xi_sym, eta_sym), sp.Matrix(derivs_2d_sym_dict[alpha]), "numpy")
                     for alpha in multi_indices}

    return shape_lambda, deriv_lambdas

# Example usage
if __name__ == "__main__":
    n = 2  # Polynomial order
    max_deriv_order = 3  # Up to third-order derivatives
    shape_fn, deriv_fns = tri_pn(n, max_deriv_order)

    xi, eta = 0.25, 0.25
    print("Shape functions:", shape_fn(xi, eta).ravel())
    print("d/dxi:", deriv_fns[(1, 0)](xi, eta).ravel())
    print("d/deta:", deriv_fns[(0, 1)](xi, eta).ravel())
    print("d^2/dxi^2:", deriv_fns[(2, 0)](xi, eta).ravel())
    print("d^2/dxi deta:", deriv_fns[(1, 1)](xi, eta).ravel())
    print("d^3/dxi^2 deta:", deriv_fns[(2, 1)](xi, eta).ravel())