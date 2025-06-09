"""
Order-agnostic reference-element factory.
"""
from functools import lru_cache
from importlib import import_module
import numpy as np

@lru_cache(maxsize=None)
def get_reference(element_type: str, poly_order: int = 1):
    if element_type == "quad":
        s_l, g_l_tpl, h_l_tpl, lap_l  = import_module("pycutfem.fem.reference.quad_qn").quad_qn(poly_order)
    elif element_type == "tri":
        s_l, g_l_tpl, h_l_tpl, lap_l = import_module("pycutfem.fem.reference.tri_pn").tri_pn(poly_order)
    else:
        raise KeyError(element_type)

    class Ref:
        @staticmethod
        def shape(xi, eta): return s_l(xi, eta).astype(float).ravel() # (n_loc,)

        # Gradient methods
        @staticmethod
        def grad_dxi(xi, eta): return g_l_tpl[0](xi, eta).astype(float).ravel() # (n_loc,)
        
        @staticmethod
        def grad_deta(xi, eta): return g_l_tpl[1](xi, eta).astype(float).ravel() # (n_loc,)

        @staticmethod
        def grad(xi, eta): # Returns (n_loc, 2)
            # Reconstructs the (n_loc, 2) matrix where each row is (dNi/dxi, dNi/deta)
            # This matches the previous Ref.grad behavior if needed for compatibility
            dphi_dxi_vals = g_l_tpl[0](xi, eta).astype(float) # Shape (n_loc, 1)
            dphi_deta_vals = g_l_tpl[1](xi, eta).astype(float) # Shape (n_loc, 1)
            return np.hstack((dphi_dxi_vals, dphi_deta_vals)) # Shape (n_loc, 2)

        # Hessian component methods
        @staticmethod
        def hess_dxixi(xi, eta): return h_l_tpl[0](xi, eta).astype(float).ravel() # (n_loc,)
        
        @staticmethod
        def hess_detaeta(xi, eta): return h_l_tpl[1](xi, eta).astype(float).ravel() # (n_loc,)
        
        @staticmethod
        def hess_dxieta(xi, eta): return h_l_tpl[2](xi, eta).astype(float).ravel() # (n_loc,)

        # Laplacian method
        @staticmethod
        def laplacian(xi, eta): return lap_l(xi, eta).astype(float).ravel() # (n_loc,)
    return Ref

