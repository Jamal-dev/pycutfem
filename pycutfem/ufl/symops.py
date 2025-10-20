# This code would live in a new file, e.g., ufl/symbolic_ops.py

import sympy
from typing import List, Tuple
from pycutfem.ufl.expressions import *
import functools



class SymbolicOps:
    """
    A utility class that provides a library of FEM-like mathematical operations
    for creating weak form integrands directly within the SymPy symbolic environment.
    
    All methods are static, allowing them to be called directly from the class,
    e.g., `SymbolicOps.dot(u, v)`, without needing to create an instance.
    """
    
    # --- 1. Coordinate System Definition ---
    # Define the base coordinate symbols once as class attributes for 2D problems.
    x, y = sympy.symbols('x y')
    _coords = (x, y)

    # --- 2. Field Creation Methods ---
    @staticmethod
    def scalar_field(name: str) -> sympy.Function:
        """
        Creates a sympy.Function representing a scalar field, e.g., p(x,y).

        Args:
            name: The base name for the field (e.g., 'p', 'q_test').
        
        Returns:
            A SymPy Function object.
        """
        return sympy.Function(name)(*SymbolicOps._coords)

    @staticmethod
    def vector_field(name: str) -> sympy.Matrix:
        """
        Creates a sympy.Matrix representing a 2D vector field.

        Args:
            name: The base name for the field (e.g., 'u_trial'). The components
                  will be named automatically (e.g., 'u_trial_x', 'u_trial_y').
        
        Returns:
            A 2x1 SymPy Matrix object.
        """
        field_x = sympy.Function(f"{name}_x")(*SymbolicOps._coords)
        field_y = sympy.Function(f"{name}_y")(*SymbolicOps._coords)
        return sympy.Matrix([field_x, field_y])

    # --- 3. Differential Operator Methods ---
    @staticmethod
    def grad(field):
        """
        Computes the symbolic gradient.
        - If the input is a scalar field, returns the gradient vector.
        - If the input is a vector field, returns the gradient tensor (Jacobian matrix).
        """
        if isinstance(field, sympy.Matrix): # Gradient of a vector -> Matrix
            return sympy.Matrix([
                [field[0].diff(c) for c in SymbolicOps._coords],
                [field[1].diff(c) for c in SymbolicOps._coords]
            ])
        else: # Gradient of a scalar -> Vector
            return sympy.Matrix([field.diff(c) for c in SymbolicOps._coords])

    @staticmethod
    def div(vector_field: sympy.Matrix) -> sympy.Expr:
        """Computes the symbolic divergence of a vector field."""
        if not isinstance(vector_field, sympy.Matrix) or vector_field.shape != (2, 1):
             raise TypeError("Divergence operator requires a 2x1 SymPy Matrix.")
        return vector_field[0].diff(SymbolicOps.x) + vector_field[1].diff(SymbolicOps.y)
        
    @staticmethod
    def curl(vector_field: sympy.Matrix) -> sympy.Expr:
        """Computes the scalar curl of a 2D vector field: d(v_y)/dx - d(v_x)/dy."""
        if not isinstance(vector_field, sympy.Matrix) or vector_field.shape != (2, 1):
            raise TypeError("Curl operator requires a 2x1 SymPy Matrix.")
        return vector_field[1].diff(SymbolicOps.x) - vector_field[0].diff(SymbolicOps.y)

    # --- 4. Algebraic Operator Methods ---
    @staticmethod
    def dot(a, b) -> sympy.Expr:
        """Computes the symbolic dot product of two vectors."""
        if isinstance(a, sympy.Matrix) and isinstance(b, sympy.Matrix):
            if a.shape == (2,1) and b.shape == (2,1): return a.dot(b) # vec.vec
            if a.shape == (2,2) and b.shape == (2,1): return a*b     # tensor.vec
            if a.shape == (2, 1) and b.shape == (2, 2):
                return b.T * a  # vec.tensor
        # Fallback for scalar multiplication
        if not isinstance(a, sympy.Matrix) or not isinstance(b, sympy.Matrix):
            return a * b
        raise TypeError(f"Unsupported dot product between shapes {a.shape} and {b.shape}")

    @staticmethod
    def cross(u, v) -> sympy.Expr:
        """Computes the scalar cross product of two 2D vectors: u_x*v_y - u_y*v_x."""
        if not all(isinstance(arg, sympy.Matrix) and arg.shape == (2, 1) for arg in [u, v]):
            raise TypeError("Cross product requires two 2x1 SymPy Matrices.")
        return u[0]*v[1] - u[1]*v[0]

    @staticmethod
    def inner(A, B) -> sympy.Expr:
        """
        Computes the symbolic inner product (Frobenius norm, A:B) of two tensors.
        """
        if not all(isinstance(arg, sympy.Matrix) and arg.shape == (2, 2) for arg in [A, B]):
            raise TypeError("Inner product requires two 2x2 SymPy Matrices.")
        # The trace of (A.T * B) is equivalent to the double contraction A:B
        return (A.T * B).trace()

    @staticmethod
    def compute_gateaux_derivative(residual: sympy.Expr, solution_vars: List[Tuple]) -> sympy.Expr:
        """
        Computes the Gâteaux derivative (the Jacobian) of a residual automatically.

        Args:
            residual: A SymPy expression for the weak form residual, R.
            solution_vars: A list of tuples, where each tuple contains the
                           solution variable (e.g., u_k) and its corresponding
                           perturbation/direction (e.g., du).

        Returns:
            A SymPy expression for the Jacobian integrand, J.
        """
        tau = sympy.Symbol('__tau__')
        
        # Create the substitution list: u_k -> u_k + tau*du, etc.
        substitutions = []
        for sol_var, pert_var in solution_vars:
            # Handle both scalar and vector fields
            if isinstance(sol_var, sympy.Matrix):
                for i in range(len(sol_var)):
                    substitutions.append((sol_var[i], sol_var[i] + tau * pert_var[i]))
            else:
                substitutions.append((sol_var, sol_var + tau * pert_var))
        
        # Substitute u -> u + tau*du into the residual
        residual_perturbed = residual.subs(substitutions)
        
        # Differentiate with respect to tau
        jacobian = sympy.diff(residual_perturbed, tau)
        
        # Evaluate at tau = 0 to get the final Jacobian
        return jacobian.subs(tau, 0)


# ---------------------------------------------------------------------------
# Sympy to UFL-like expressions
# ---------------------------------------------------------------------------
class SymPyToUFLVisitor:
    """
    Translates a SymPy expression tree into a UFL expression tree.
    """
    def __init__(self, symbol_map: dict):
        """
        Initializes the visitor with a map from SymPy Functions to UFL Functions.
        Example: {sympy.Function('u_trial_x')(x,y): ufl.TrialFunction('ux')}
        """
        self.symbol_map = symbol_map
        # More robust mapping for derivative coordinates
        self.coord_map = {sympy.Symbol('x'): 0, sympy.Symbol('y'): 1}


    def visit(self, node):
        """
        Public visit method for dispatching. This version uses isinstance
        for robust type checking.
        """
        # The order matters: check for more specific types first if they exist.
        if isinstance(node, sympy.Derivative):
            return self.visit_Derivative(node)
        if isinstance(node, sympy.Add):
            return self.visit_Add(node)
        if isinstance(node, sympy.Mul):
            return self.visit_Mul(node)
        # This is the key fix: It catches all symbolic functions like
        # 'du_trial_x', 'p_k', etc., because they are all instances of sympy.Function.
        if isinstance(node, sympy.Function):
            return self.visit_Function(node)
        if isinstance(node, sympy.Pow):
            return self.visit_Pow(node)
        if isinstance(node, sympy.Integer):
            return self.visit_Integer(node)
        if isinstance(node, sympy.Float):
            return self.visit_Float(node)
        if isinstance(node, sympy.Rational):
            return self.visit_Rational(node)
        if isinstance(node, sympy.Symbol):
            return self.visit_Symbol(node)

        # Fallback for any type not handled above
        return self.generic_visit(node)

    def generic_visit(self, node):
        raise TypeError(f"No UFL translation rule for SymPy object: {type(node)} with value {node}")

    # --- Translation Rules for SymPy Object Types ---

    def visit_Add(self, node):
        """Translates a SymPy Add into a tree of UFL Sum objects."""
        terms = [self.visit(arg) for arg in node.args]
        return functools.reduce(lambda a, b: Sum(a, b), terms)

    def visit_Mul(self, node):
        """Translates a SymPy Mul into a tree of UFL Prod objects."""
        terms = [self.visit(arg) for arg in node.args]
        return functools.reduce(lambda a, b: Prod(a, b), terms)

    def visit_Pow(self, node):
        """Translates a SymPy Pow, e.g., x**2."""
        base = self.visit(node.base)
        if node.exp == -1: return Prod(Constant(1.0), Div(Constant(1.0), base)) # Placeholder for 1/x
        raise NotImplementedError("General ufl.Pow not implemented yet.")

    def visit_Integer(self, node):
        """Translates a SymPy Integer to a UFL Constant."""
        return Constant(int(node))

    def visit_Float(self, node):
        """Translates a SymPy Float to a UFL Constant."""
        return Constant(float(node))
        
    def visit_Rational(self, node):
        """Translates a SymPy Rational to a UFL Constant."""
        return Constant(float(node))

    def visit_Symbol(self, node):
        """
        Translates a SymPy Symbol.
        It first attempts to find the symbol in the provided map.
        This is crucial for replacing symbols like 'theta' or 'mu' with
        their corresponding UFL Constant objects that hold numerical values.
        """
        # First, check if there's a direct mapping for this symbol.
        if node in self.symbol_map:
            # If so, return the mapped UFL object (e.g., the Constant(0.5) object).
            return self.symbol_map[node]

        # This should not be reached for physical constants if the map is correct.
        raise ValueError(
            f"Encountered a free SymPy symbol '{node.name}' that was not found "
            f"in the symbol_map. All symbolic constants (like rho, mu, dt) "
            f"must be explicitly mapped to UFL Constant objects."
        )

    def visit_Function(self, node):
        """Translates a symbolic function using the provided symbol map."""
        if node in self.symbol_map:
            return self.symbol_map[node]
        raise ValueError(f"Unknown SymPy function '{node}' not found in symbol map.")

    def visit_Derivative(self, node):
        """Translates a SymPy derivative into a UFL Grad component."""
        field_to_diff = self.visit(node.expr)
        
        # Determine if differentiating w.r.t 'x' (0) or 'y' (1)
        coord_symbol = node.variables[0]
        if coord_symbol not in self.coord_map:
            raise ValueError(f"Differentiating with respect to an unknown coordinate: {coord_symbol}")
            
        coord_index = self.coord_map[coord_symbol]
        return Grad(field_to_diff)[coord_index]

