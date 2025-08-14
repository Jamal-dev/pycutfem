# pycutfem.fem.reference
"""
Order-agnostic reference-element factory.
"""
from functools import lru_cache
from importlib import import_module
import numpy as np

class Ref:
    def __init__(self, shape_lambda, deriv_lambdas):
        self.shape_lambda = shape_lambda
        self.deriv_lambdas = deriv_lambdas

    @lru_cache(maxsize=None)
    def shape(self, xi, eta):
        return self.shape_lambda(xi, eta).astype(float).ravel()

    @lru_cache(maxsize=None)
    def derivative(self, xi, eta, order_xi, order_eta):
        alpha = (order_xi, order_eta)
        if alpha not in self.deriv_lambdas:
            raise ValueError(f"Derivative order {alpha} not computed. "
                             f"Adjust max_deriv_order >= {order_xi + order_eta}.")
        return self.deriv_lambdas[alpha](xi, eta).astype(float).ravel()

    # Compatibility methods
    @lru_cache(maxsize=None)
    def grad_dxi(self, xi, eta):
        return self.derivative(xi, eta, 1, 0)

    @lru_cache(maxsize=None)
    def grad_deta(self, xi, eta):
        return self.derivative(xi, eta, 0, 1)

    @lru_cache(maxsize=None)
    def grad(self, xi, eta):
        dphi_dxi = self.grad_dxi(xi, eta)
        dphi_deta = self.grad_deta(xi, eta)
        return np.hstack((dphi_dxi[:, None], dphi_deta[:, None]))

    @lru_cache(maxsize=None)
    def hess_dxixi(self, xi, eta):
        return self.derivative(xi, eta, 2, 0)

    @lru_cache(maxsize=None)
    def hess_detaeta(self, xi, eta):
        return self.derivative(xi, eta, 0, 2)

    @lru_cache(maxsize=None)
    def hess_dxieta(self, xi, eta):
        return self.derivative(xi, eta, 1, 1)

    @lru_cache(maxsize=None)
    def laplacian(self, xi, eta):
        return self.hess_dxixi(xi, eta) + self.hess_detaeta(xi, eta)
    @lru_cache(maxsize=None)
    def hess(self, xi, eta):
        d20 = self.hess_dxixi(xi, eta)
        d11 = self.hess_dxieta(xi, eta)
        d02 = self.hess_detaeta(xi, eta)
        H = np.empty((d20.shape[0], 2, 2), dtype=float)
        H[:, 0, 0] = d20
        H[:, 0, 1] = d11
        H[:, 1, 0] = d11
        H[:, 1, 1] = d02
        return H

@lru_cache(maxsize=None)
def get_reference(element_type: str, poly_order: int = 1, max_deriv_order: int = 2):
    if element_type == "quad":
        shape_l, deriv_lambdas  = import_module("pycutfem.fem.reference.quad_qn").quad_qn(poly_order, max_deriv_order)
    elif element_type == "tri":
        shape_l, deriv_lambdas = import_module("pycutfem.fem.reference.tri_pn").tri_pn(poly_order, max_deriv_order)
    else:
        raise KeyError(element_type)

                
    return Ref(shape_l, deriv_lambdas)

