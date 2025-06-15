from functools import lru_cache
import sympy as sp
import numpy as np
from itertools import product

@lru_cache(maxsize=None)
def _lagrange_basis_1d(n: int, max_deriv_order: int):
    """
    Generates 1D Lagrange basis functions and their derivatives up to max_deriv_order.
    
    Args:
        n: Polynomial order (number of nodes is n + 1).
        max_deriv_order: Maximum derivative order to compute.
    
    Returns:
        tuple: (basis_sym, derivs_sym, x_sym, nodes_1d)
            basis_sym: List of symbolic 1D basis functions L_i(x).
            derivs_sym: List of lists, derivs_sym[k] contains kth derivatives of all basis functions.
            x_sym: Symbolic variable sp.Symbol("x").
            nodes_1d: NumPy array of 1D node locations on [-1, 1].
    """
    x_sym = sp.Symbol("x")
    nodes_1d = np.linspace(-1, 1, n + 1)
    
    basis_sym = []
    derivs_sym = [[] for _ in range(max_deriv_order + 1)]  # 0th to max_deriv_order derivatives
    
    for i, xi_node in enumerate(nodes_1d):
        li_sym = sp.S(1)
        for j, xj_node in enumerate(nodes_1d):
            if i != j:
                li_sym *= (x_sym - xj_node) / (xi_node - xj_node)
        
        simplified_li = sp.simplify(li_sym)
        basis_sym.append(simplified_li)
        
        current_deriv = simplified_li
        derivs_sym[0].append(current_deriv)
        for order in range(1, max_deriv_order + 1):
            current_deriv = sp.simplify(sp.diff(current_deriv, x_sym))
            derivs_sym[order].append(current_deriv)
    
    return basis_sym, derivs_sym, x_sym, nodes_1d

@lru_cache(maxsize=None)
def quad_qn(n: int, max_deriv_order: int = 2):
    """
    Return lambdified shape functions and derivatives up to max_deriv_order for Qn elements.
    
    Args:
        n: Polynomial order.
        max_deriv_order: Maximum derivative order (default 2 for Hessian).
    
    Returns:
        tuple: (shape_lambda, deriv_lambdas)
            shape_lambda: Callable returning shape function values (n_basis_funcs,).
            deriv_lambdas: Dict with multi-index keys (alpha_xi, alpha_eta), values are callables
                           returning derivative values (n_basis_funcs,).
    """
    L_1d_sym, derivs_1d_sym, x1d_sym, _ = _lagrange_basis_1d(n, max_deriv_order)
    xi_sym, eta_sym = sp.symbols("xi eta")
    n_basis = (n + 1) * (n + 1)
    
    # Generate multi-indices up to max_deriv_order
    multi_indices = []
    for total_order in range(max_deriv_order + 1):
        for alpha_xi in range(total_order + 1):
            alpha_eta = total_order - alpha_xi
            multi_indices.append((alpha_xi, alpha_eta))
    
    # Store symbolic derivatives
    basis_2d_sym_list = []
    derivs_2d_sym_dict = {alpha: [] for alpha in multi_indices}
    
    for j in range(n + 1):  # eta direction
        for i in range(n + 1):  # xi direction
            Li_xi = L_1d_sym[i].subs({x1d_sym: xi_sym})
            Lj_eta = L_1d_sym[j].subs({x1d_sym: eta_sym})
            phi_sym = sp.simplify(Li_xi * Lj_eta)
            basis_2d_sym_list.append(phi_sym)
            
            for alpha_xi, alpha_eta in multi_indices:
                dLi_dxi = derivs_1d_sym[alpha_xi][i].subs({x1d_sym: xi_sym})
                dLj_deta = derivs_1d_sym[alpha_eta][j].subs({x1d_sym: eta_sym})
                dphi = sp.simplify(dLi_dxi * dLj_deta)
                derivs_2d_sym_dict[(alpha_xi, alpha_eta)].append(dphi)
    
    # Lambdify shape functions
    shape_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(basis_2d_sym_list), "numpy")
    
    # Lambdify derivatives
    deriv_lambdas = {}
    for alpha in multi_indices:
        deriv_matrix = sp.Matrix(derivs_2d_sym_dict[alpha])
        deriv_lambdas[alpha] = sp.lambdify((xi_sym, eta_sym), deriv_matrix, "numpy")
    
    return shape_lambda, deriv_lambdas

# Example usage in __init__.py would need adjustment; here’s a minimal test
if __name__ == "__main__":
    shape_fn, deriv_fns = quad_qn(2, max_deriv_order=3)
    xi, eta = 0.5, -0.5
    print("Shape:", shape_fn(xi, eta).ravel())
    print("d/dxi:", deriv_fns[(1, 0)](xi, eta).ravel())
    print("d^3/dxi^2deta:", deriv_fns[(2, 1)](xi, eta).ravel())
    # print("d^4/dxi^2deta^2:", deriv_fns[(2, 2)](xi, eta).ravel())