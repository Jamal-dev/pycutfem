# pycutfem/fem/compilers.py
"""Light‑yet‑robust *local* form compiler for the project’s test-suite.

It supports exactly the expression types exercised by the Stokes examples
(Q2/Q1 Taylor–Hood, vector formulation) **plus** a generic interior‑facet path
so future ghost‑penalty terms won’t break again.  All duplicated definitions
that crept into the file have been removed – this is the *only* `FormCompiler`
exported.

Key design points
-----------------
* **Side‑aware basis cache**  – visitors pick the right trace when the context
  advertises an `active_side` (test) or `trial_side`.
* **Volume and facet assembly paths share the same visitors**; only the way
  `basis_values` is filled differs.
* **No dependence on a `level_set`** for ordinary `interior_facet` measures; if
  present, we use it to orient the normal consistently (+ side has φ ≥ 0).
* **RHS assembly** is sign‑aware via `_split_terms`.

The implementation aims for clarity over absolute speed, but all hot loops are
NumPy‑vectorised and free of Python allocations.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Mapping, Any, Union
import logging,functools

# UFL‑like helpers ----------------------------------------------------------
from ufl.expressions import (
    Constant, TrialFunction, TestFunction,
    VectorTrialFunction, VectorTestFunction,
    Grad, DivOperation, Inner, Dot, Sum, Sub, Prod,
    Jump, FacetNormal, Pos, Neg, Function, VectorFunction, Div
)
from ufl.forms import Equation

# Project helpers -----------------------------------------------------------
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.reference import get_reference
from pycutfem.fem import transform
from pycutfem.integration import volume
from pycutfem.integration.quadrature import  line_quadrature
from pycutfem.integration.quadrature import edge as edge_quadrature_element
import sympy

_INTERFACE_TOL = 1.0e-12



# ---------------------------------------------------------------------------
#  Light shape‑meta visitor (helps flip test/trial in ∇· terms)
# ---------------------------------------------------------------------------
class _Shape:
    __slots__ = ("dim", "kind")
    def __init__(self, dim: int, kind: str):
        self.dim = dim      # tensor order (scalar:0, vector:1, grad:2,…)
        self.kind = kind    # 'test' | 'trial' | 'none'



# In ufl/compilers.py

class _ShapeVisitor:
    def __init__(self):
        self._memo = {}
    def visit(self, n):
        if n in self._memo: return self._memo[n]
        
        # This dispatch needs to be specific
        meth_name = f"visit_{type(n).__name__}"
        meth = getattr(self, meth_name, self.generic_visit)
        
        res = meth(n)
        self._memo[n] = res
        return res

    def generic_visit(self, n):
        # Fallback for base classes if a specific visitor doesn't exist
        if isinstance(n, TestFunction): return self.visit_TestFunction(n)
        if isinstance(n, TrialFunction): return self.visit_TrialFunction(n)
        if isinstance(n, Function): return self.visit_Function(n)
        raise TypeError(f"No shape rule for {type(n)}")

    # --- Corrected Visitors for all types ---
    def visit_Constant(self, n):      return _Shape(n.dim, 'none')
    def visit_Analytic(self, n):      return _Shape(0, 'none')
    
    # --- Specific handlers for each type ---
    def visit_TestFunction(self, n):      return _Shape(0, 'test')
    def visit_TrialFunction(self, n):     return _Shape(0, 'trial')
    def visit_Function(self, n):          return _Shape(0, 'none')
    
    def visit_VectorTestFunction(self, n):  return _Shape(1, 'test')
    def visit_VectorTrialFunction(self, n): return _Shape(1, 'trial')
    def visit_VectorFunction(self, n):      return _Shape(1, 'none')

    # --- Visitors for Operators ---
    def visit_Sum(self, n):  return self.visit(n.a)
    def visit_Sub(self, n):  return self.visit(n.a)
    def visit_Prod(self,n):
        sa, sb = self.visit(n.a), self.visit(n.b)
        dim = sa.dim + sb.dim
        kind = sa.kind if sa.kind != 'none' else sb.kind
        return _Shape(dim, kind)

    def visit_Grad(self, n): 
        s = self.visit(n.operand)
        return _Shape(s.dim + 1, s.kind)

    def visit_DivOperation(self,n):
        return _Shape(0, self.visit(n.operand).kind)

    def visit_Inner(self,n): 
        sa = self.visit(n.a); sb = self.visit(n.b)
        kind = sa.kind if sa.kind != 'none' else sb.kind
        return _Shape(max(0, sa.dim - 1), kind)
    
    def visit_Dot(self,n):
        sa = self.visit(n.a); sb = self.visit(n.b)
        kind = sa.kind if sa.kind != 'none' else sb.kind
        # Dot product reduces total tensor rank by 2, or dimension by 1 for vectors
        dim = max(0, sa.dim + sb.dim - 2) if (sa.dim > 0 and sb.dim > 0) else sa.dim + sb.dim
        return _Shape(dim, kind)

    def visit_Div(self, n): return self.visit(n.a)
    def visit_Pos(self,n):   return self.visit(n.operand)
    def visit_Neg(self,n):   return self.visit(n.operand)
    def visit_Jump(self,n):  return self.visit(n.u_pos)
    def visit_FacetNormal(self,n): return _Shape(1,'none')
    def visit_ElementWiseConstant(self, n):
        return _Shape(0, 'none')

# ---------------------------------------------------------------------------
#  Helper utilities
# ---------------------------------------------------------------------------

def _split_terms(expr):
    """Flatten a +/- expression tree into (sign, term) pairs."""
    if isinstance(expr, Sum):
        return _split_terms(expr.a) + _split_terms(expr.b)
    if isinstance(expr, Sub):
        return _split_terms(expr.a) + [(-s, t) for s, t in _split_terms(expr.b)]
    return [(1, expr)]


def _all_fields(expr):
    fields=set()
    def walk(n):
        if hasattr(n,'field_name'):
            fields.add(n.field_name)
        if hasattr(n,'space'):
            fields.update(n.space.field_names)
        if isinstance(n, VectorFunction):
            fields.update(n.field_names)

        # Recurse through the expression tree
        for attr in ('operand', 'a', 'b', 'u_pos', 'u_neg', 'components','f'):
            if hasattr(n, attr):
                m = getattr(n, attr)
                if isinstance(m, (list, tuple)):
                    for x in m: walk(x)
                elif m is not None:
                    walk(m)
    walk(expr)
    return list(fields)


def _trial_test(expr):
    trial = expr.find_first(lambda n:isinstance(n,(TrialFunction,VectorTrialFunction)))
    test  = expr.find_first(lambda n:isinstance(n,(TestFunction ,VectorTestFunction )))
    return trial, test

# ---------------------------------------------------------------------------
#  Actual compiler
# ---------------------------------------------------------------------------
class FormCompiler:
    def __init__(self, dofhandler: DofHandler, quad_order: int | None = None, assembler_hooks: dict | None = None):
        self.dh = dofhandler
        self.qorder = quad_order
        self.shape = _ShapeVisitor()
        self.ctx: Dict[str,Any] = {}
        # assembler hooks are passed in via `ctx['hooks']`
        self.ctx['hooks'] = assembler_hooks or {}
        # bucket for diagnostics such as perimeter / jump integrals
        self._scalar_results: dict[str, float] = {}
        self.ctx['scalar_results'] = self._scalar_results

    # ---------- public --------------------------------------------------
    def assemble(self, equation: Equation, bcs):
        n = self.dh.total_dofs
        K = sp.lil_matrix((n,n))
        F = np.zeros(n)

        # left‑hand side
        self.ctx['is_rhs'] = False
        self._assemble_form(equation.a, K)
        # right‑hand side
        self.ctx['is_rhs'] = True
        self._assemble_form(equation.L, F)

        self._apply_bcs(K,F,bcs)
        # face-only forms (e.g. Constant()*ds) produce no matrix/vector
        # – in this case hand back the accumulated scalars instead
        if self._scalar_results:
           return self._scalar_results
        return K.tocsr(), F

    # ---------- visitors ------------------------------------------------
    def visit(self, node):
        return getattr(self, f"visit_{type(node).__name__}")(node)

    def visit_ElementWiseConstant(self, node):
        side = self.ctx.get("side", "")
        if side:
            elem_id = self.ctx["e_pos"] if side == "+" else self.ctx["e_neg"]
        else:
            elem_id = self.ctx["elem_id"]
        return node.values[elem_id]
        
    # ---------------- Derivative single ------------------------------------------
    def visit_Derivative(self, node):
        """
        Evaluates a single component of a gradient.
        e.g., Derivative(f, 0) corresponds to ∂f/∂x.

        This method is robust enough to handle both 2D arrays of basis
        gradients (from Trial/TestFunctions) and 1D arrays of interpolated
        gradient values (from data-carrying Functions).
        """
        # First, evaluate the full gradient of the function.
        full_grad = self.visit(Grad(node.f))
        component_idx = node.component_index

        # --- NEW: Check the dimensionality of the result ---

        if full_grad.ndim == 2:
            # This is the case for Trial/TestFunctions, where full_grad is an
            # array of basis gradients with shape (n_basis, n_dim).
            # We need to select the correct column.
            return full_grad[:, component_idx]
        
        elif full_grad.ndim == 1:
            # This is the case for a data-carrying Function, where full_grad
            # is the final interpolated gradient vector with shape (n_dim,).
            # We just need to select the correct component.
            return full_grad[component_idx]
        
        else:
            # Handle scalars or other unexpected shapes
            if full_grad.ndim == 0 and component_idx == 0:
                return full_grad
            raise ValueError(f"Unsupported shape {full_grad.shape} in visit_Derivative")

    # scalars
    def visit_Constant(self,n): 
        v = n.value
        if n.dim ==0: # scalar constant
            if isinstance(v, (int, float)):
                return v
            if isinstance(v, np.ndarray) and v.ndim == 0:
                return v.item()
            if callable(v):  # e.g. lambda x,y: x
                if 'x' not in self.ctx:
                    raise RuntimeError("Coordinate-dependent Constant evaluated outside quadrature loop")
                x, y = self.ctx['x']
                v = v(x, y)  # evaluate at quadrature point
        elif n.dim == 1:  # vector constant
            if callable(v[0]):
                if 'x' not in self.ctx:
                    raise RuntimeError("Coordinate-dependent Constant evaluated outside quadrature loop")
                x, y = self.ctx['x']
                v = np.array([f(x, y) for f in v])
            if isinstance(v, List):
                return np.array(v, dtype=float)
        return v
    
    def visit_Pos(self, n):
        """
        Evaluates the operand if on the positive side (phi>=0), otherwise returns
        a zero of the same shape as the operand.
        """
        # print("visit Pos zeroing: ctx=", self.ctx)
        # We use a clear convention: '+' side includes the boundary (phi >= 0)
        if ('phi_val' in self.ctx and self.ctx['phi_val'] >= _INTERFACE_TOL) or \
           ('active_side' in self.ctx and self.ctx['active_side'] != '+'):
            
            # Get operand shape and return a correctly-shaped zero
            sh = self.shape.visit(n.operand)
            if sh.dim == 0: return 0.0
            if sh.dim == 1: return np.zeros(2) # Assumes 2D vectors
            # Add other dimensions if needed
            raise TypeError(f"Unsupported dimension {sh.dim} in Pos/Neg zeroing.")

        return self.visit(n.operand)

    def visit_Neg(self, n):
        """
        Evaluates the operand if on the negative side (phi<0), otherwise returns
        a zero of the same shape as the operand.
        """
        # print("visit Neg zeroing: ctx=", self.ctx)
        # The '-' side is strictly phi < 0
        if ('phi_val' in self.ctx and self.ctx['phi_val'] < _INTERFACE_TOL) or \
           ('active_side' in self.ctx and self.ctx['active_side'] != '-'):

            # Get operand shape and return a correctly-shaped zero
            sh = self.shape.visit(n.operand)
            if sh.dim == 0: return 0.0
            if sh.dim == 1: return np.zeros(2) # Assumes 2D vectors
            raise TypeError(f"Unsupported dimension {sh.dim} in Pos/Neg zeroing.")
            
        return self.visit(n.operand)
    def visit_Sum(self,n): return self.visit(n.a)+self.visit(n.b)
    def visit_Sub(self,n): return self.visit(n.a)-self.visit(n.b)

    # basis look‑up helpers ------------------------------------------
    def _basis_val(self, field:str, role:str):
        """role: 'test'|'trial'; uses ctx['active_side'] / ctx['trial_side'] if present"""
        bv = self.ctx['basis_values']
        if '+' in bv:   # side‑aware dict
            side = self.ctx['active_side'] if role=='test' else self.ctx.get('trial_side','+')
            return bv[side][field]['val']
        return bv[field]['val']

    def _basis_grad(self, field:str, role:str):
        bv = self.ctx['basis_values']
        if '+' in bv:
            side = self.ctx['active_side'] if role=='test' else self.ctx.get('trial_side','+')
            return bv[side][field]['grad']
        return bv[field]['grad']

    # finite element functions ------------------------------------------
    def visit_TestFunction(self,n):  return self._basis_val(n.field_name,'test')
    def visit_TrialFunction(self,n): return self._basis_val(n.field_name,'trial')

    def visit_VectorTestFunction(self,n):  return np.hstack([self._basis_val(f,'test')  for f in n.space.field_names])
    def visit_VectorTrialFunction(self,n): return np.hstack([self._basis_val(f,'trial') for f in n.space.field_names])

    #----------------------------------------------------------
    def _get_data_vector_and_dofs(self, func_object):
        """
        Unified helper to get the correct data vector and element DoFs.
        It checks for a solver context and falls back to the object's own data.
        """
        elem_dofs = self._elm_dofs(func_object, self.ctx['elem_id'])
        
        if 'solution_vector' in self.ctx:
            # --- SOLVER MODE ---
            solution_vectors = self.ctx['solution_vector']
            
            # In solver mode, a VectorFunction's components might not be in the context dict.
            # We use the base name (e.g., 'u_k' from 'u_k_ux') if it exists.
            name_to_lookup = getattr(func_object, '_parent_vector', func_object).name
            
            data_vector = solution_vectors.get(name_to_lookup)
            if data_vector is None:
                raise KeyError(f"Function '{name_to_lookup}' not found in solution_vectors context.")
            
            # In solver mode, we slice the GLOBAL vector directly with GLOBAL dofs.
            nodal_loc = data_vector[elem_dofs]
            
        else:
            # --- DIRECT EVALUATION MODE ---
            # Call the object's own getter method, which handles the mapping
            # from global element DoFs to its internal local data array.
            nodal_loc = func_object.get_nodal_values(elem_dofs)
            
        return nodal_loc

    def visit_Function(self, n):
        elem_dofs = self._elm_dofs(n, self.ctx['elem_id'])
        nodal_loc = n.get_nodal_values(elem_dofs)
        N = self._basis_val(n.field_name, role='test')
        return N @ nodal_loc
    
    def visit_VectorFunction(self, n):
        # This method is for when a VectorFunction is evaluated directly, e.g. dot(beta, ...).
        # Its logic is more complex as it combines multiple fields.
        field_values = []
        for component in n.components:
            # We can just visit each component as a regular Function
            field_values.append(self.visit(component))
        return np.array(field_values)

    # differential ops ---------------------------------------------------
    def visit_Grad(self, n):
        # This branch handles known data functions.
        if isinstance(n.operand, (Function, VectorFunction)) and not isinstance(n.operand, (TrialFunction, TestFunction)):
            func = n.operand
            if isinstance(func, Function): # Scalar case
                elem_dofs = self._elm_dofs(func, self.ctx['elem_id'])
                nodal_loc = func.get_nodal_values(elem_dofs)
                G = self._basis_grad(func.field_name, role='test')
                return G.T @ nodal_loc
            else: # Vector case
                grad_rows = []
                for component in func.components:
                    elem_dofs = self._elm_dofs(component, self.ctx['elem_id'])
                    nodal_loc_comp = component.get_nodal_values(elem_dofs)
                    G = self._basis_grad(component.field_name, role='test')
                    grad_rows.append(G.T @ nodal_loc_comp)
                return np.vstack(grad_rows)

        # This branch for symbolic functions remains unchanged
        sh = self.shape.visit(n.operand)
        if sh.dim == 0:
            return self._basis_grad(n.operand.field_name, sh.kind)
        if sh.dim == 1:
            return [self._basis_grad(f, sh.kind) for f in n.operand.space.field_names]
        raise NotImplementedError('grad of tensor not needed')

    def visit_DivOperation(self,n):
        grads = self.visit(Grad(n.operand))  # list of rows
        return np.hstack([g[:,i] for i,g in enumerate(grads)])
    
    def visit_Analytic(self, node): return node.eval(self.ctx['x_phys'])

    # algebra -----------------------------------------------------------
    def visit_Prod(self,n):
        a = self.visit(n.a); b=self.visit(n.b)
        if self.ctx['is_rhs']:
            return a*b
        sa,sb = self.shape.visit(n.a), self.shape.visit(n.b)
        if sa.kind=='none' or sb.kind=='none':
            return a*b
        # orient outer‑product row=test, col=trial
        if sa.kind=='test' and sb.kind=='trial':
            return np.outer(a,b)
        if sb.kind=='test' and sa.kind=='trial':
            return np.outer(b,a)
        if sa.kind == 'none' and sb.kind == 'none':
            return a * b
        # Handle convection: e.g., dot(du, grad(u_k)) * v
        if sa.kind == 'none' and sb.kind == 'test':  # dot(du, grad(u_k)) * v
            return np.outer(b, a)  # v as rows, dot result as cols (needs du basis)
        raise TypeError(f"Unsupported Prod: sa={sa.kind}, sb={sb.kind}")
    
    def visit_Div(self, n):
        """Handles scalar, vector, or tensor division by a scalar."""
        numerator = self.visit(n.a)
        denominator = self.visit(n.b)

        # NumPy correctly handles element-wise division of an array by a scalar.
        return numerator / denominator

    def visit_Inner(self,n):
        a=self.visit(n.a); b=self.visit(n.b)
        sa,sb = self.shape.visit(n.a), self.shape.visit(n.b)
        if sa.dim==2 and sb.dim==2:  # grad(u):grad(v)
            rows = np.cumsum([0]+[c.shape[0] for c in a])
            cols = np.cumsum([0]+[c.shape[0] for c in b])
            mat = np.zeros((rows[-1], cols[-1]))
            for i,(ra,rb) in enumerate(zip(a,b)):
                mat[rows[i]:rows[i+1], cols[i]:cols[i+1]] = ra @ rb.T
            return mat
        # ADDED: This new path correctly handles grad(scalar).grad(scalar) for Poisson.
        if sa.dim == 1 and sb.dim == 1:
            # `a` and `b` are matrices of basis gradients, shape (n_basis, n_dim).
            # We compute G_v @ G_u.T to get the (n_basis, n_basis) matrix.
            # `b` is from the test function (rows), `a` is from the trial function (cols).
            return b @ a.T
        # scalar inner
        if sa.kind=='test':
            return np.outer(a,b)
        return np.outer(b,a)


    # dot product ---------------------------------------------------
    #----------------------------------------------------------------------------------------------------------

    def visit_Dot(self, n):
        """
        Compute the dot product between two operands in a finite element context.
        Handles Stokes load vectors, advection terms, numerical arrays, and other FEM cases.
        """
        # Visit operands unless they are symbolic expressions to be handled directly
        a = self.visit(n.a) if not isinstance(n.a, (Constant, Grad, Jump, FacetNormal, TrialFunction, VectorTrialFunction, VectorFunction)) else n.a
        b = self.visit(n.b) if not isinstance(n.b, (Constant, Grad, Jump, FacetNormal, TestFunction, VectorTestFunction, VectorFunction)) else n.b

        # Case 1: Numerical arrays (e.g., after evaluation)
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if a.ndim == 1 and b.ndim == 1:
                return np.dot(a, b)  # Vector dot product
            elif a.ndim == 2 and b.ndim == 1:
                return a @ b  # Matrix-vector product
            elif a.ndim == 0 or b.ndim == 0:
                return a * b  # Scalar multiplication
            else:
                raise TypeError(f"Unsupported shapes in dot: {a.shape}, {b.shape}")

        # Case 2: Stokes load vector - dot(Constant, VectorTestFunction) or vice versa
        if (isinstance(n.a, Constant) and isinstance(n.b, VectorTestFunction)) or \
        (isinstance(n.b, Constant) and isinstance(n.a, VectorTestFunction)):
            const = n.a if isinstance(n.a, Constant) else n.b
            test = n.b if isinstance(n.b, VectorTestFunction) else n.a
            const_val = const.value  # Constant vector, e.g., [fx, fy]
            test_val = self.visit(test)  # DOFs of vector test function, shape (n_dofs_total,)

            ncomp = len(test.space.field_names)  # Number of components (e.g., 2 for 2D)
            if len(const_val) != ncomp:
                raise ValueError("Constant vector dimension must match test function components")
            if len(test_val) % ncomp != 0:
                raise ValueError("Test function DOFs must be divisible by number of components")
            
            dofs_per_comp = len(test_val) // ncomp
            out = np.zeros_like(test_val)
            for i in range(ncomp):
                start = i * dofs_per_comp
                end = (i + 1) * dofs_per_comp
                out[start:end] = const_val[i] * test_val[start:end]  # Scale each component
            return out

        # Case 3: Advection term - dot(Constant, Grad(TrialFunction))
        if isinstance(n.a, Constant) and isinstance(n.b, Grad) and isinstance(n.b.operand, TrialFunction):
            const_val = n.a.value  # Constant vector, e.g., beta = [1, 1]
            grad_trial = self.visit(n.b)  # Gradient matrix, shape (n_basis, n_dim)
            if not isinstance(grad_trial, np.ndarray):
                return Dot(n.a, n.b)  # Defer if symbolic
            if len(const_val) != grad_trial.shape[1]:
                raise ValueError("Dimension mismatch in dot(Constant, Grad)")
            return np.dot(grad_trial, const_val)  # Resulting shape: (n_basis,)

        # Case 4: Boundary/flux term - dot(Grad(...), FacetNormal())
        if isinstance(n.a, Grad) and isinstance(n.b, FacetNormal):
            grad_val = self.visit(n.a)  # Gradient, e.g., (n_basis, n_dim)
            normal = self.visit(n.b)  # Normal vector, e.g., (n_dim,)
            if isinstance(grad_val, np.ndarray) and isinstance(normal, np.ndarray):
                if grad_val.ndim == 2:
                    return np.dot(grad_val, normal)  # (n_basis,)
                elif grad_val.ndim == 1:
                    return np.dot(grad_val, normal)  # Scalar
            return Dot(n.a, n.b)  # Defer symbolic case

        # Case 5: DG term - dot(Jump(...), FacetNormal())
        if isinstance(n.a, Jump) and isinstance(n.b, FacetNormal):
            jump_val = self.visit(n.a)  # Jump across facets
            normal = self.visit(n.b)  # Normal vector
            if isinstance(jump_val, np.ndarray) and isinstance(normal, np.ndarray):
                return np.dot(jump_val, normal)
            return Dot(n.a, n.b)  # Defer symbolic case

        # Case 6: Scalar test function - dot(Constant, TestFunction) or vice versa
        if (isinstance(n.a, Constant) and isinstance(n.b, TestFunction)) or \
        (isinstance(n.b, Constant) and isinstance(n.a, TestFunction)):
            const = n.a if isinstance(n.a, Constant) else n.b
            test = n.b if isinstance(n.b, TestFunction) else n.a
            const_val = const.value
            test_val = self.visit(test)
            if not isinstance(const_val, (int, float)):
                raise ValueError("Constant must be scalar for scalar TestFunction")
            return const_val * test_val
        
        # Handles terms like dot(u_k, grad(du_i)) from the advection term.
        if isinstance(n.a, VectorFunction) and isinstance(n.b, Grad) and isinstance(n.b.operand, TrialFunction):
            vec_func_val = self.visit(n.a)    # Evaluated vector, e.g., u_k at a point -> shape(2,)
            grad_trial = self.visit(n.b)      # Basis gradients for du_i -> shape(n_basis, 2)
            
            # This is the operation u_k ⋅ ∇(du_i), which results in a vector of
            # coefficients for each basis function in the trial space.
            return grad_trial @ vec_func_val  # (n_basis, 2) @ (2,) -> (n_basis,)

        # Case 7: Trial and test functions (bilinear forms)
        a_is_trial = isinstance(n.a, (TrialFunction, VectorTrialFunction))
        b_is_test = isinstance(n.b, (TestFunction, VectorTestFunction))
        if a_is_trial and b_is_test:
            if isinstance(n.a, VectorTrialFunction) and isinstance(n.b, VectorTestFunction):
                # Vector case: sum over components
                return sum(self.visit_Prod(Prod(n.a[i], n.b[i])) for i in range(len(n.a.space.field_names)))
            return self.visit_Prod(Prod(n.a, n.b))  # Scalar case

        # If no case matches, raise an error
        print(n.a.dim, n.b.shape)
        raise NotImplementedError(f"Dot between {type(n.a)} and {type(n.b)} not supported")

    # def visit_Dot(self, n): # this logic is passihng the Stokes test
    #     # --- Dispatcher: Correctly identify the expression pattern ---

    #     # Use `isinstance` to check the UFL types directly
    #     is_a_const = isinstance(n.a, Constant)
    #     is_b_const = isinstance(n.b, Constant)
    #     is_a_test = isinstance(n.a, (TestFunction, VectorTestFunction))
    #     is_b_test = isinstance(n.b, (TestFunction, VectorTestFunction))

    #     # If the pattern is `dot(Constant, TestFunction)`, use the special Stokes logic
    #     if (is_a_const and is_b_test) or (is_b_const and is_a_test):
    #         const_eval, vec_eval = (self.visit(n.a), self.visit(n.b)) if is_a_const else (self.visit(n.b), self.visit(n.a))

    #         ncomp = len(const_eval)
    #         if len(vec_eval) == 0:
    #             return vec_eval  # Return empty if no DoFs on element

    #         # Ensure component division is valid
    #         if len(vec_eval) % ncomp != 0:
    #             raise ValueError(
    #                 f"In dot(Constant, TestFunction), cannot divide basis vector of "
    #                 f"length {len(vec_eval)} into {ncomp} components."
    #             )
            
    #         dofs_per = len(vec_eval) // ncomp
    #         out = np.zeros_like(vec_eval)
    #         vec_reshaped = vec_eval.reshape((ncomp, dofs_per))
    #         for i in range(ncomp):
    #             out[i * dofs_per:(i + 1) * dofs_per] = const_eval[i] * vec_reshaped[i]
    #         return out

    #     # --- General Logic Path ---
    #     # If the Stokes pattern doesn't match, proceed with general-purpose logic.
    #     a = self.visit(n.a)
    #     b = self.visit(n.b)

    #     a = np.asarray(a)
    #     b = np.asarray(b)

    #     # Case for dot(vector, vector) -> scalar (fixes the jump test)
    #     if a.ndim == 1 and b.ndim == 1:
    #         return np.dot(a, b)
            
    #     # Case for scalar * vector/matrix
    #     if a.ndim == 0: return a * b
    #     if b.ndim == 0: return b * a

    #     # ADDED: Case for dot(tensor, vector) -> vector (Matrix-vector product)
    #     if a.ndim == 2 and b.ndim == 1:
    #         return a @ b
            
    #     # # Failing the advection diffusion test: Case for dot(vector, tensor) -> vector
    #     # if a.ndim == 1 and b.ndim == 2:
    #     #     return a @ b

    #     # Other cases for assembling vectors or matrices
    #     sa, sb = self.shape.visit(n.a), self.shape.visit(n.b)
    #     # ADDED: This new condition handles dot(Constant, grad(TrialFunction))
    #     if sa.kind == 'none' and sb.kind == 'trial':
    #         # This is Constant `a` dotted with each basis grad in `b`.
    #         # Operation is matrix @ vector: (4, 2) @ (2,) -> (4,)
    #         return b @ a
    #     if sa.kind == 'none' and sb.kind == 'test': return b @ a
    #     if sa.kind == 'test' and sb.kind == 'none': return a @ b.T
    #     if sa.kind == 'test' and sb.kind == 'trial': return np.outer(a, b)
    #     if sb.kind == 'test' and sa.kind == 'trial': return np.outer(b, a)
            
    #     raise TypeError(f"Unsupported Dot operation between shapes {a.shape} and {b.shape}")

    #---------------------------------------------------------------------------------------------------------

    # def visit_Jump(self,n):
    #     side = self.ctx['active_side']
    #     if side=='+': return self.visit(n.u_pos)
    #     return self.visit(n.u_neg)

    # def visit_Jump(self, n):                         # u(+) – u(–)
    #     return self.visit(n.u_pos) - self.visit(n.u_neg)
    
    def visit_Jump(self, n):
        """
        Correct evaluation on CutFEM interfaces:
        we must obtain *both* traces, regardless of the current φ-sign.
        """
        phi_orig = self.ctx.get('phi_val', None)

        # ----- “+” trace -------------------------------------------------
        # Temporarily force phi to be positive to ensure the Pos() visitor
        # evaluates its operand.
        if phi_orig is not None:
            self.ctx['phi_val'] = +1.0
        u_pos = self.visit(n.u_pos)

        # ----- “–” trace -------------------------------------------------
        # Temporarily force phi to be negative to ensure the Neg() visitor
        # evaluates its operand.
        if phi_orig is not None:
            self.ctx['phi_val'] = -1.0
        u_neg = self.visit(n.u_neg)

        # ----- restore context ------------------------------------------
        # Restore the original phi_val to prevent side-effects.
        if phi_orig is None:
            self.ctx.pop('phi_val', None)
        else:
            self.ctx['phi_val'] = phi_orig

        return u_pos - u_neg

    def visit_FacetNormal(self,n):
        return self.ctx['normal']

    # ---------- core assembly helpers -----------------------------------
    def _apply_bcs(self,K,F,bcs):
        data = self.dh.get_dirichlet_data(bcs)
        if not data: return
        rows = np.fromiter(data.keys(),int)
        vals = np.fromiter(data.values(),float)
        u = np.zeros_like(F); u[rows]=vals
        F -= K@u
        K = K.tolil()
        K[rows,:]=0; K[:,rows]=0; K[rows,rows]=1.0
        F[rows]=vals

    def _assemble_form(self,form,matvec):
        # A RHS like “Constant(0.0)” has no .measure; just ignore it
        if not hasattr(form, "integrals"):
            return

        from ufl.measures import Integral 
        for integral in form.integrals:
            if not isinstance(integral, Integral):
                continue                          # skip stray Constants

            kind = integral.measure.domain_type
            if kind=='volume':
                self._assemble_volume(integral,matvec)
            elif kind=='interior_facet':
                self._assemble_facet(integral,matvec)
            elif kind=='interface':
                self._assemble_interface(integral,matvec)

    # volume ------------------------------------------------------------
    # In ufl/compilers.py

    def _assemble_volume(self, intg, matvec):
        rhs = self.ctx['is_rhs']
        # Use the first field to determine the mesh for geometry, assuming they are all on the same one.
        mesh = self.dh.fe_map[_all_fields(intg.integrand)[0]]
        q = self.qorder or mesh.poly_order + 2
        qpts, qwts = volume(mesh.element_type, q)
        terms = _split_terms(intg.integrand)
        
        # Get all unique fields in the integrand to get their reference elements
        all_q_fields = _all_fields(intg.integrand)
        ref_elements = {fld: get_reference(self.dh.fe_map[fld].element_type, self.dh.fe_map[fld].poly_order)
                        for fld in all_q_fields}

        for eid, elem in enumerate(mesh.elements_list):
            self.ctx['elem_id'] = elem.id

            for sgn, term in terms:
                trial, test = _trial_test(term)
                if rhs and test is None: continue
                if not rhs and (trial is None or test is None): continue
                
                row = self._elm_dofs(test, eid)
                if rhs:
                    local = np.zeros(len(row))
                else:
                    col = self._elm_dofs(trial, eid)
                    local = np.zeros((len(row), len(col)))

                # Main Quadrature Loop
                for xi_eta, w in zip(qpts, qwts):
                    # --- All calculations are now correctly scoped inside this loop ---

                    # 1. Compute geometric transformations for the CURRENT quadrature point
                    J = transform.jacobian(mesh, eid, xi_eta)
                    detJ = abs(np.linalg.det(J))
                    JinvT = np.linalg.inv(J).T
                    
                    # 2. Set the context for the visit methods
                    self.ctx['x_phys'] = transform.x_mapping(mesh, elem.id, xi_eta)
                    self.ctx['x'] = self.ctx['x_phys']
                    self.ctx["side"] = ""

                    # 3. Compute and cache basis function values for the CURRENT quadrature point
                    basis_values_at_qp = {}
                    for fld in _all_fields(term): # Only need fields for the specific term
                        ref = ref_elements[fld]
                        basis_values_at_qp[fld] = {
                            'val': ref.shape(*xi_eta),
                            'grad': ref.grad(*xi_eta) @ JinvT
                        }
                    self.ctx['basis_values'] = basis_values_at_qp

                    # 4. Evaluate the integrand
                    val = self.visit(term)
                    
                    # 5. Accumulate into the local matrix/vector
                    local += sgn * w * detJ * val

                # Scatter the local contribution into the global matrix/vector
                if rhs:
                    matvec[row] += local
                else:
                    matvec[np.ix_(row, col)] += local
                    
        # Context cleanup (good practice)
        for key in ('phi_val', 'normal', 'basis_values', 'x', 'nodal_vals', 'elem_id', 'x_phys', 'side'):
            self.ctx.pop(key, None)
    # def _assemble_volume(self,intg,matvec):
    #     rhs = self.ctx['is_rhs']
    #     mesh = self.dh.fe_map[_all_fields(intg.integrand)[0]]
    #     q = self.qorder or mesh.poly_order+2
    #     qpts,qwts = volume(mesh.element_type,q)
    #     terms = _split_terms(intg.integrand)
    #     for eid,elem in enumerate(mesh.elements_list):
            
    #         self.ctx['elem_id'] = elem.id
    #         # precompute basis at qpts
    #         bv=defaultdict(lambda:{'val':[],'grad':[]})
    #         for fld in _all_fields(intg.integrand):
    #             fm = self.dh.fe_map[fld]
    #             ref = get_reference(fm.element_type,fm.poly_order)
    #             for xi,eta in qpts:
    #                 J = transform.jacobian(fm,eid,(xi,eta))
    #                 JinvT = np.linalg.inv(J).T
    #                 bv[fld]['val'].append(ref.shape(xi,eta))
    #                 bv[fld]['grad'].append(ref.grad(xi,eta) @ JinvT)
    #         for sgn,term in terms:
    #             trial,test=_trial_test(term)
    #             if rhs and test is None: continue
    #             if not rhs and (trial is None or test is None): continue
    #             row = self._elm_dofs(test,eid)
    #             if rhs:
    #                 local = np.zeros(len(row))
    #             else:
    #                 col = self._elm_dofs(trial,eid)
    #                 local = np.zeros((len(row),len(col)))
    #             for k,(xi_eta, w) in enumerate(zip(qpts,qwts)):
    #                 self.ctx['x_phys'] = transform.x_mapping(mesh, elem.id, (xi, eta))
    #                 self.ctx["side"] = ""  # volume integrals have no side notion
    #                 self.ctx['x'] = self.ctx['x_phys']
    #                 self.ctx['basis_values']={f:{'val':bv[f]['val'][k],'grad':bv[f]['grad'][k]} for f in bv}
    #                 J = transform.jacobian(mesh,eid,xi_eta)
    #                 val = self.visit(term)
    #                 local += sgn * w * abs(np.linalg.det(J)) * val
    #             if rhs:
    #                 matvec[row]+=local
    #             else:
    #                 matvec[np.ix_(row,col)]+=local
    #     # Context cleanup
    #     for key in ('phi_val', 'normal', 'basis_values', 'x', 'nodal_vals','elem_id','x_phys','side'):
    #         self.ctx.pop(key, None)
    
    # ------------------------------------------------------------------
    #  Interface integrals (CutFEM) – one-sided, no extra DOFs
    # ------------------------------------------------------------------
    # In class FormCompiler:

    def _assemble_interface(self, intg, matvec):
        """
        Assembles integrals over non-conforming interfaces defined by a level set.

        This robust implementation evaluates the entire integrand at once, correctly
        handling scalar and vector-valued expressions for bilinear forms, linear
        forms, and hooked functionals without relying on term-splitting.
        """
        # Use a logger for clear, hierarchical debugging output.
        # (Assumes you have `import logging` at the top of the file)
        log = logging.getLogger(__name__)
        log.debug(f"Assembling interface integral: {intg}")

        # --- 1. Initial Setup ---
        rhs = self.ctx['is_rhs']
        level_set = intg.measure.level_set
        if level_set is None:
            raise ValueError("dInterface measure requires a level_set.")
            
        mesh = next(iter(self.dh.fe_map.values())) # Any mesh known to the DoF handler
        qdeg = self.qorder or mesh.poly_order + 2
        fields = _all_fields(intg.integrand)
        
        # Check for a user-defined hook for functionals.
        hook = self.ctx['hooks'].get(type(intg.integrand))
        if hook and hook.get('name') not in self.ctx.get('scalar_results', {}):
            # Initialize the result bucket; it might hold a scalar or a vector.
            self.ctx.setdefault('scalar_results', {})[hook['name']] = 0.0

        try:
            # --- 2. Loop Over All Cut Elements in the Mesh ---
            for elem in mesh.elements_list:
                if elem.tag != 'cut' or len(elem.interface_pts) != 2:
                    continue
                
                self.ctx['elem_id'] = elem.id
                log.debug(f"  Processing cut element: {elem.id}")

                # --- 3. Quadrature Rule on the Physical Interface Segment ---
                p0, p1 = elem.interface_pts
                qp, qw = line_quadrature(p0, p1, qdeg)

                # --- 4. Pre-compute Basis Functions for this Element ---
                # This is a key optimization. We evaluate all basis functions for all
                # quadrature points on this element's interface segment at once.
                basis_cache = {}
                if fields:
                    for fld in fields:
                        fm = self.dh.fe_map[fld]
                        ref = get_reference(fm.element_type, fm.poly_order)
                        vals, grads = [], []
                        for x_q in qp:
                            xi, eta = transform.inverse_mapping(fm, elem.id, x_q)
                            J = transform.jacobian(fm, elem.id, (xi, eta))
                            # The mathematically correct transformation for row-vector gradients
                            Jinv = np.linalg.inv(J)
                            vals.append(ref.shape(xi, eta))
                            grads.append(ref.grad(xi, eta) @ Jinv)
                        # Store as efficient NumPy arrays
                        basis_cache[fld] = {'val': np.asarray(vals), 'grad': np.asarray(grads)}
                
                # --- 5. Determine Assembly Path (Bilinear, Linear, or Functional) ---
                trial, test = _trial_test(intg.integrand)

                # --- PATH A: Bilinear Form (e.g., a(u,v), contributes to the matrix) ---
                if not rhs and trial and test:
                    row_dofs = self._elm_dofs(test, elem.id)
                    col_dofs = self._elm_dofs(trial, elem.id)
                    loc = np.zeros((len(row_dofs), len(col_dofs)))
                    
                    for k, (x, w) in enumerate(zip(qp, qw)):
                        # Set context for the current quadrature point
                        self.ctx['x'], self.ctx['phi_val'] = x, level_set(x)
                        self.ctx['normal'] = level_set.gradient(x)
                        if fields:
                            self.ctx['basis_values'] = {f: {'val': basis_cache[f]['val'][k], 'grad': basis_cache[f]['grad'][k]} for f in fields}
                        
                        # Evaluate the ENTIRE integrand at once
                        integrand_val = self.visit(intg.integrand)
                        loc += w * integrand_val
                    
                    matvec[np.ix_(row_dofs, col_dofs)] += loc
                    log.debug(f"    Assembled {loc.shape} local matrix for element {elem.id}")

                # --- PATH B: Linear Form (e.g., L(v)) or Functional (e.g., dot(jump,n)) ---
                else:
                    # Initialize accumulator correctly based on the result shape
                    acc = None
                    
                    # Main Quadrature Loop
                    for k, (x, w) in enumerate(zip(qp, qw)):
                        self.ctx['x'], self.ctx['phi_val'] = x, level_set(x)
                        self.ctx['normal'] = level_set.gradient(x)
                        if fields:
                            self.ctx['basis_values'] = {f: {'val': basis_cache[f]['val'][k], 'grad': basis_cache[f]['grad'][k]} for f in fields}
                        
                        integrand_val = self.visit(intg.integrand)
                        
                        # Initialize accumulator on the first pass
                        if acc is None:
                            acc = w * np.asarray(integrand_val)
                        else:
                            acc += w * integrand_val

                    # Add the accumulated result to the correct global object
                    if acc is not None:
                        if rhs: # It's a linear form (RHS)
                            row_dofs = self._elm_dofs(test, elem.id)
                            matvec[row_dofs] += acc
                            log.debug(f"    Assembled {acc.shape} local vector for element {elem.id}")
                        elif hook: # It's a hooked functional
                            self.ctx['scalar_results'][hook['name']] += acc
                            log.debug(f"    Accumulated functional '{hook['name']}' for element {elem.id}")

        finally:
            # --- 6. Final Context Cleanup ---
            # Use a `finally` block to guarantee cleanup happens even if an error occurs.
            for key in ('phi_val', 'normal', 'basis_values', 'x', 'elem_id'):
                self.ctx.pop(key, None)
            log.debug("Interface assembly finished. Context cleaned.")




    # facet -------------------------------------------------------------
    def _assemble_facet(self,intg,matvec):
        rhs = self.ctx['is_rhs']
         # ----- pick *any* mesh the dof-handler knows if the integrand
        #       contains no FE fields (e.g. Constant * ds)
        fields = _all_fields(intg.integrand)
        mesh   = self.dh.fe_map[fields[0]] if fields else next(iter(self.dh.fe_map.values()))

        # optional user-defined hook, e.g. {Constant: {'name':'perimeter'}}
        hooks = self.ctx.get('hooks', {})          # <-- hooks were stored here
        scalar_hook = None
        for cls, cfg in hooks.items():             # accept subclass matches
            if isinstance(intg.integrand, cls):
                scalar_hook = cfg
                break

        if scalar_hook and scalar_hook['name'] not in self.ctx['scalar_results']:
            self.ctx['scalar_results'][scalar_hook['name']] = 0.0
        q = self.qorder or mesh.poly_order+2
        edge_ids = intg.measure.defined_on.to_indices()
        terms = _split_terms(intg.integrand)
        level_set = getattr(intg.measure,'level_set',None)
        print(f"level_set={level_set}")

        for eid_edge in edge_ids:
            edge = mesh.edge(eid_edge)
            left,right = edge.left, edge.right
            if right is None:  # boundary facet – treat right as ghost copy of left
                continue
                left,right = (edge.left, edge.left)
            # decide pos/neg orientation using level_set if available
            if level_set is not None:
                phi_left = level_set(mesh.elements_list[left].centroid())
                pos,neg = (left,right) if phi_left>=0 else (right,left)
            else:
                pos,neg = left,right
            # quadrature pts on reference edge of pos element
            loc_idx = mesh.elements_list[pos].edges.index(eid_edge)
            glob_edge_idx = mesh.elements_list[pos].edges[loc_idx]
            # print(loc_idx,mesh.elements_list[pos].edges)
            qpts_ref, qwts = edge_quadrature_element(mesh.element_type, loc_idx, q)
            for qp, w in zip(qpts_ref,qwts):
                xq = transform.x_mapping(mesh,pos,qp)
                self.ctx['normal'] = mesh.edges_list[glob_edge_idx].calc_normal_unit_vector()#transform.edge_unit_normal(mesh,pos,loc_idx)  # project helper must exist
                jac1d = transform.jacobian_1d(mesh,pos,qp,loc_idx)
                # ----- Constant / jump integrals – no basis evaluation needed
                if scalar_hook and not fields:
                    val = self.visit(intg.integrand)
                    self._scalar_results[scalar_hook['name']] += val * jac1d * w
                    continue
                # side‑aware basis cache
                bv={'+' : defaultdict(dict), '-' : defaultdict(dict)}
                for side,eid in (('+',pos),('-',neg)):
                    for fld in _all_fields(intg.integrand):
                        fm = self.dh.fe_map[fld]
                        ref = get_reference(fm.element_type,fm.poly_order)
                        xi,eta = transform.inverse_mapping(fm,eid,xq)
                        J = transform.jacobian(fm,eid,(xi,eta))
                        JinvT = np.linalg.inv(J).T
                        bv[side][fld]['val']=ref.shape(xi,eta)
                        bv[side][fld]['grad']=ref.grad(xi,eta) @ JinvT
                self.ctx['basis_values']=bv

                for sgn,term in terms:
                    trial,test=_trial_test(term)
                    if rhs and test is None: continue
                    if not rhs and (trial is None or test is None): continue

                    for row_side,row_eid in (('+',pos),('-',neg)):
                        self.ctx['active_side']=row_side
                        rows = self._elm_dofs(test,row_eid)
                        if rhs:
                            val = self.visit(term)
                            matvec[rows]+= sgn*w*jac1d*val
                        else:
                            for col_side,col_eid in (('+',pos),('-',neg)):
                                self.ctx['trial_side']=col_side
                                cols = self._elm_dofs(trial,col_eid)
                                val = self.visit(term)
                                matvec[np.ix_(rows,cols)]+= sgn*w*jac1d*val

                self.ctx.pop('active_side',None)
                self.ctx.pop('trial_side',None)

    # dof helpers --------------------------------------------------------
    def _elm_dofs(self,func,eid):
        if isinstance(func,(VectorTrialFunction,VectorTestFunction)):
            return [d for f in func.space.field_names for d in self.dh.element_maps[f][eid]]
        return self.dh.element_maps[func.field_name][eid]
