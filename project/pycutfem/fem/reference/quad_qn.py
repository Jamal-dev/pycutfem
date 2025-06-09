"""
pycutfem.fem.reference.quad_qn
Tensor-product Qn basis (any n≥1) on [-1,1]×[-1,1].
"""
from functools import lru_cache
import sympy as sp
import numpy as np

@lru_cache(maxsize=None)
def _lagrange_basis_1d(n: int):
    """
    Generates 1D Lagrange basis functions, their first and second derivatives.
    
    Returns:
        tuple: (basis_sym, d_basis_sym, dd_basis_sym, x_sym, nodes_1d)
            basis_sym: List of symbolic 1D basis functions L_i(x).
            d_basis_sym: List of symbolic 1D first derivatives L_i'(x).
            dd_basis_sym: List of symbolic 1D second derivatives L_i''(x).
            x_sym: The symbolic variable sp.Symbol("x").
            nodes_1d: NumPy array of 1D node locations on [-1, 1].
    """
    x_sym = sp.Symbol("x")
    nodes_1d = np.linspace(-1, 1, n + 1)
    
    basis_sym = []
    d_basis_sym = []
    dd_basis_sym = []
    
    for i, xi_node in enumerate(nodes_1d):
        li_sym = sp.S(1) # Start with SymPy's representation of 1
        for j, xj_node in enumerate(nodes_1d):
            if i != j:
                li_sym *= (x_sym - xj_node) / (xi_node - xj_node)
        
        simplified_li = sp.simplify(li_sym)
        basis_sym.append(simplified_li)
        d_basis_sym.append(sp.simplify(sp.diff(simplified_li, x_sym)))
        dd_basis_sym.append(sp.simplify(sp.diff(simplified_li, x_sym, 2)))
        
    return basis_sym, d_basis_sym, dd_basis_sym, x_sym, nodes_1d

@lru_cache(maxsize=None)
def quad_qn(n: int):
    """
    Return lambdified (shape, grad_components, hess_components, laplacian)
    callables for Qn (quadrilateral Lagrange) elements of order n.

    grad_components: Tuple (dPhi_dxi_lambda, dPhi_deta_lambda)
    hess_components: Tuple (d2Phi_dxixi_lambda, d2Phi_detaeta_lambda, d2Phi_dxieta_lambda)
    Each lambda function returns a vector of values for all basis functions.
    """
    L_1d_sym, dL_1d_sym, ddL_1d_sym, x1d_sym, _ = _lagrange_basis_1d(n)
    xi_sym, eta_sym = sp.symbols("xi eta")

    basis_2d_sym_list = []
    # For gradients (component-wise for each basis function)
    dPhi_dxi_sym_list = []
    dPhi_deta_sym_list = []
    # For Hessian components (component-wise for each basis function)
    d2Phi_dxixi_sym_list = []
    d2Phi_detaeta_sym_list = []
    d2Phi_dxieta_sym_list = []
    # For Laplacians (for each basis function)
    laplacian_Phi_sym_list = []

    # The loops ensure a lexicographical ordering (eta changes slowest, then xi)
    # This matches typical node numbering for Qn elements.
    for Lj_eta in L_1d_sym:        # Basis function for eta direction (slowest index)
        for Li_xi in L_1d_sym:    # Basis function for xi direction (fastest index)
            # Substitute symbols and simplify
            phi_sym = sp.simplify(Li_xi.subs({x1d_sym: xi_sym}) * Lj_eta.subs({x1d_sym: eta_sym}))
            basis_2d_sym_list.append(phi_sym)

            # Find corresponding 1D derivatives for this Li_xi and Lj_eta
            # This assumes L_1d_sym, dL_1d_sym, ddL_1d_sym are ordered lists
            # and Li_xi / Lj_eta can be found by value. A safer way might be to iterate with indices.
            # However, since these are unique symbolic expressions from leggauss for a given n, .index() should be reliable.
            idx_i = -1
            for k_idx, item_k in enumerate(L_1d_sym):
                if item_k == Li_xi: # Comparing Sympy expressions
                    idx_i = k_idx
                    break
            
            idx_j = -1
            for k_idx, item_k in enumerate(L_1d_sym):
                if item_k == Lj_eta:
                    idx_j = k_idx
                    break
            
            if idx_i == -1 or idx_j == -1:
                raise ValueError("Symbolic 1D basis function not found in list during Qn construction.")


            dLi_dxi_sym = dL_1d_sym[idx_i].subs({x1d_sym: xi_sym})
            ddLi_dxixi_sym = ddL_1d_sym[idx_i].subs({x1d_sym: xi_sym})
            
            dLj_deta_sym = dL_1d_sym[idx_j].subs({x1d_sym: eta_sym})
            ddLj_detaeta_sym = ddL_1d_sym[idx_j].subs({x1d_sym: eta_sym})

            # Substitute original 1D basis functions for eta/xi parts
            Li_xi_substituted = Li_xi.subs({x1d_sym: xi_sym})
            Lj_eta_substituted = Lj_eta.subs({x1d_sym: eta_sym})

            # Gradients: d(Li(xi)Lj(eta))/dxi = dLi/dxi * Lj(eta)
            dphi_dxi = sp.simplify(dLi_dxi_sym * Lj_eta_substituted)
            # Gradients: d(Li(xi)Lj(eta))/deta = Li(xi) * dLj/deta
            dphi_deta = sp.simplify(Li_xi_substituted * dLj_deta_sym)
            dPhi_dxi_sym_list.append(dphi_dxi)
            dPhi_deta_sym_list.append(dphi_deta)

            # Hessian components
            ddphi_dxixi = sp.simplify(ddLi_dxixi_sym * Lj_eta_substituted) # d2(Li)/dxi2 * Lj
            ddphi_detaeta = sp.simplify(Li_xi_substituted * ddLj_detaeta_sym) # Li * d2(Lj)/deta2
            ddphi_dxieta = sp.simplify(dLi_dxi_sym * dLj_deta_sym) # (dLi/dxi) * (dLj/deta)
            d2Phi_dxixi_sym_list.append(ddphi_dxixi)
            d2Phi_detaeta_sym_list.append(ddphi_detaeta)
            d2Phi_dxieta_sym_list.append(ddphi_dxieta)
            
            # Laplacian (w.r.t. reference coordinates xi, eta)
            laplacian_phi = sp.simplify(ddphi_dxixi + ddphi_detaeta)
            laplacian_Phi_sym_list.append(laplacian_phi)

    # Lambdify
    # For shape: input (xi,eta), output is a COLUMN vector (n_basis_funcs x 1)
    shape_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(basis_2d_sym_list), "numpy")
    
    # For gradients: each returns a COLUMN vector (n_basis_funcs x 1)
    dPhi_dxi_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(dPhi_dxi_sym_list), "numpy")
    dPhi_deta_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(dPhi_deta_sym_list), "numpy")

    # For Hessian components: each returns a COLUMN vector (n_basis_funcs x 1)
    d2Phi_dxixi_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(d2Phi_dxixi_sym_list), "numpy")
    d2Phi_detaeta_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(d2Phi_detaeta_sym_list), "numpy")
    d2Phi_dxieta_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(d2Phi_dxieta_sym_list), "numpy")

    # For Laplacians: returns a COLUMN vector (n_basis_funcs x 1)
    laplacian_lambda = sp.lambdify((xi_sym, eta_sym), sp.Matrix(laplacian_Phi_sym_list), "numpy")
    
    return (shape_lambda,
            (dPhi_dxi_lambda, dPhi_deta_lambda),
            (d2Phi_dxixi_lambda, d2Phi_detaeta_lambda, d2Phi_dxieta_lambda),
            laplacian_lambda)
