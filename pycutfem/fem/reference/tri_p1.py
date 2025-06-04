"""pycutfem.fem.reference.tri_p1"""
import sympy as sp
xi,eta=sp.symbols('xi eta')
N_sym=sp.Matrix([1-xi-eta, xi, eta])
dN_sym=N_sym.jacobian([xi,eta])
shape=sp.lambdify((xi,eta),N_sym,'numpy')
grad=sp.lambdify((xi,eta),dN_sym,'numpy')
