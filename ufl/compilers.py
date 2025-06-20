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

def _split_terms(expr, coef=1.0):
    """
    Flatten a ± tree **and** pull purely-scalar factors in front.
    Returns a list  [(scalar, term), …]
    where each *term* still contains the symbolic FE factors.
    """
    from ufl.expressions import Sum, Sub, Prod, Constant
    # 1. walk through +/–
    if isinstance(expr, Sum):
        return _split_terms(expr.a, coef) + _split_terms(expr.b, coef)
    if isinstance(expr, Sub):
        return _split_terms(expr.a, coef) + _split_terms(expr.b, -coef)

    # 2. pull scalar constants out of a product
    if isinstance(expr, Prod):
        # “Constant ⋅ something”  or  “something ⋅ Constant”
        if isinstance(expr.a, Constant) and expr.a.dim == 0:
            return _split_terms(expr.b, coef * expr.a.value)
        if isinstance(expr.b, Constant) and expr.b.dim == 0:
            return _split_terms(expr.a, coef * expr.b.value)

    # 3. everything else stays as-is
    return [(coef, expr)]




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
        """
        Correct evaluation of gradients for both scalar‐ and vector-valued
        data functions and for symbolic (test / trial) functions.
        """

        # ------------------------------------------------------------------ #
        # 1. Concrete FE functions (u_k, u_n, beta, …) ---------------------- #
        # ------------------------------------------------------------------ #
        if isinstance(n.operand, (Function, VectorFunction)) \
        and not isinstance(n.operand, (TrialFunction, TestFunction)):

            func       = n.operand
            nodal_loc  = self._get_data_vector_and_dofs(func)   # (dofs_elem,)

            # ---------- scalar ------------------------------------------------
            if isinstance(func, Function):
                G = self._basis_grad(func.field_name, role='test')   # (n_b, n_dim)
                return G.T @ nodal_loc                               # (n_dim,)

            # ---------- vector ------------------------------------------------
            n_comp = len(func.components)
            n_bs   = nodal_loc.size // n_comp                        # scalar bases
            grad_rows = []
            for i, comp in enumerate(func.components):
                comp_nodal = nodal_loc[i*n_bs : (i+1)*n_bs]          # slice once
                G          = self._basis_grad(comp.field_name, 'test')
                grad_rows.append(G.T @ comp_nodal)                   # (n_dim,)
            return np.vstack(grad_rows)                              # (n_comp,n_dim)

        # ------------------------------------------------------------------ #
        # 2. Symbolic (trial / test) functions ------------------------------ #
        # ------------------------------------------------------------------ #
        sh = self.shape.visit(n.operand)

        if sh.dim == 0:                    # grad(scalar ϕ) → (n_b,n_dim)
            return self._basis_grad(n.operand.field_name, sh.kind)

        if sh.dim == 1:                    # grad(vector ϕ) → (n_comp,n_b,n_dim)
            return np.stack([self._basis_grad(f, sh.kind)
                            for f in n.operand.space.field_names])

        raise TypeError(f"Unsupported operand to Grad(): {type(n.operand)}")


    def visit_DivOperation(self,n):
        grads = self.visit(Grad(n.operand))
        if isinstance(grads, np.ndarray) and grads.ndim == 3: # Symbolic
            return np.hstack([grads[i, :, i] for i in range(grads.shape[0])])
        elif isinstance(grads, np.ndarray) and grads.ndim == 2: # Data
            return np.trace(grads)
        raise TypeError(f"Unsupported type for DivOperation: {type(grads)}")
    
    def visit_Analytic(self, node): return node.eval(self.ctx['x_phys'])

    # algebra -----------------------------------------------------------
    
    
    def visit_Div(self, n):
        """Handles scalar, vector, or tensor division by a scalar."""
        numerator = self.visit(n.a)
        denominator = self.visit(n.b)

        # NumPy correctly handles element-wise division of an array by a scalar.
        return numerator / denominator

    def visit_Inner(self, n):
        """ FINAL, ROBUST visit_Inner method """
        a_node, b_node = n.a, n.b
        
        if self.ctx['is_rhs']:
            # --- RHS: Assemble a vector from inner(func, test) ---
            grad_func_val = self.visit(a_node)   # Evaluated tensor, shape (2,2)
            grad_test_basis = self.visit(b_node) # STACKED basis grads, shape (2, 9, 2)

            n_comp = grad_test_basis.shape[0]
            n_basis_scalar = grad_test_basis.shape[1]
            out = np.zeros(n_basis_scalar * n_comp)
            
            for i in range(n_comp):
                start, end = i * n_basis_scalar, (i + 1) * n_basis_scalar
                # grad_test_basis[i] is (9,2). grad_func_val[i] is (2,). Result is (9,).
                out[start:end] = grad_test_basis[i] @ grad_func_val[i]
            return out
            
        else:
            # --- LHS: Assemble a matrix from inner(trial, test) ---
            G_a = self.visit(a_node)
            G_b = self.visit(b_node)
            sa = self.shape.visit(a_node)
            
            # Reorder to ensure G_test is from the test function (row)
            G_test = G_b if sa.kind == 'trial' else G_a
            G_trial = G_a if sa.kind == 'trial' else G_b
            
            if sa.dim == 2: # inner(grad(vector), grad(vector))
                n_comp = G_test.shape[0]
                n_basis_scalar = G_test.shape[1]
                local_matrix = np.zeros((n_basis_scalar*n_comp, n_basis_scalar*n_comp))
                for i in range(n_comp):
                    block = G_test[i] @ G_trial[i].T # (9,2) @ (2,9) -> (9,9)
                    start, end = i * n_basis_scalar, (i + 1) * n_basis_scalar
                    local_matrix[start:end, start:end] = block
                return local_matrix
            else: # inner(grad(scalar), grad(scalar))
                return G_test @ G_trial.T
                
        raise NotImplementedError("Inner product form not supported for these shapes/kinds.")


    # dot product ---------------------------------------------------
    #----------------------------------------------------------------------------------------------------------
    def visit_Prod(self, n):
        """
        Calculates the product of two expressions.
        For bilinear forms (LHS), if the product involves a 'trial' kind expression
        and a 'test' kind expression, it performs an outer product to form a local
        element matrix. Otherwise, it's a simple scalar/vector multiplication.
        """
        a_val = self.visit(n.a)
        b_val = self.visit(n.b)
        
        if self.ctx['is_rhs']:
            # RHS: Simple scalar/vector multiplication for load vector contributions
            return a_val * b_val
        
        # LHS: This is where we form the stiffness/mass matrix contributions
        sa, sb = self.shape.visit(n.a), self.shape.visit(n.b)

        # Identify which operand is 'test' and which is 'trial' based on shape kind.
        # This handles:
        # - (TestFunction * TrialFunction) directly
        # - (ComplexTrialExpression * TestFunction) or (TestFunction * ComplexTrialExpression)
        #   where ComplexTrialExpression is, for example, dot(beta, grad(u[0])) which has shape.kind='trial'
        
        test_expr_val = None
        trial_expr_val = None

        if sa.kind == 'test':
            test_expr_val = a_val
        elif sb.kind == 'test':
            test_expr_val = b_val
        
        if sa.kind == 'trial':
            trial_expr_val = a_val
        elif sb.kind == 'trial':
            trial_expr_val = b_val
        
        # If we have both a 'test' component and a 'trial' component, form an outer product.
        if test_expr_val is not None and trial_expr_val is not None:
            # Ensure both are 1D arrays suitable for np.outer.
            # This is crucial for results like dot(beta, grad(u[0])) which is 1D.
            if not isinstance(test_expr_val, np.ndarray) or test_expr_val.ndim != 1 or \
               not isinstance(trial_expr_val, np.ndarray) or trial_expr_val.ndim != 1:
                raise ValueError(
                    f"Prod expects 1D arrays for outer product between 'test' ({test_expr_val.shape}) "
                    f"and 'trial' ({trial_expr_val.shape}) kinds. "
                    f"Nodes: {type(n.a).__name__} ({sa.kind}), {type(n.b).__name__} ({sb.kind})"
                )
            return np.outer(test_expr_val, trial_expr_val)
        
        if isinstance(a_val, np.ndarray) and isinstance(b_val, np.ndarray):
            # 1. scalar × anything  ----------------------------------------
            if a_val.ndim == 0:
                return a_val * b_val
            if b_val.ndim == 0:
                return b_val * a_val

            # 2. vector × vector (1-D vs 1-D) ------------------------------
            if a_val.ndim == 1 and b_val.ndim == 1:
                if self.ctx['is_rhs']:
                    # RHS → element-wise multiply keeps a load vector
                    return a_val * b_val
                else:
                    # LHS → need the outer product for a matrix block
                    return np.outer(a_val, b_val)

            # 3. matrix × vector  or  vector × matrix ----------------------
            if a_val.ndim == 2 and b_val.ndim == 1 and a_val.shape[1] == b_val.shape[0]:
                return a_val @ b_val
            if a_val.ndim == 1 and b_val.ndim == 2 and a_val.shape[0] == b_val.shape[0]:
                return a_val @ b_val

        # If not forming an outer product (e.g., Constant * Constant, Function * Function, etc.)
        # or if only one of them is a test/trial function, perform direct multiplication.
        # This covers Constant * TestFunction (for RHS when `is_rhs` is false, but typically RHS)
        # and other cases where we don't form a matrix block.
        # This is essentially the "none" kind fallback.
        return a_val * b_val
    

    def _block_sizes(self,stacked, ncomp):
        """Return `n_basis_scalar` given a stacked [ux-block | uy-block]."""
        if stacked.size % ncomp:
            raise ValueError("Vector basis length not divisible by n_components")
        return stacked.size // ncomp

    def _weight_vector_basis(self, basis_stacked, weights):
        """
        Multiply each scalar-component block of a stacked vector basis
        by the corresponding entry in `weights` (length 2 in 2-D).
        """
        ncomp = len(weights)
        nbs   = self._block_sizes(basis_stacked, ncomp)
        out   = np.empty_like(basis_stacked)
        for i, w in enumerate(weights):
            out[i*nbs : (i+1)*nbs] = w * basis_stacked[i*nbs : (i+1)*nbs]
        return out

    def _diag_outer(self, test_vec, trial_vec, ncomp):
        """
        Build the usual *block-diagonal* outer product for
        dot(VectorTestFunction, VectorTrialFunction).
        """
        nbs = self._block_sizes(test_vec, ncomp)
        M   = np.zeros((ncomp*nbs, ncomp*nbs))
        for i in range(ncomp):
            tb = test_vec [i*nbs : (i+1)*nbs]
            rb = trial_vec[i*nbs : (i+1)*nbs]
            M[i*nbs : (i+1)*nbs, i*nbs : (i+1)*nbs] = np.outer(tb, rb)
        return M
   

    def visit_Dot(self, n):
        a_node, b_node = n.a, n.b
        a_val,  b_val  = self.visit(a_node), self.visit(b_node)

        if (isinstance(a_node, TrialFunction) and isinstance(b_node, TestFunction)) or \
        (isinstance(a_node, TestFunction) and isinstance(b_node, TrialFunction)):

            # Make sure we know which one is the row (test) and which is the column (trial)
            test_basis  = self.visit(b_node if isinstance(b_node, TestFunction)  else a_node)  # 1-D (n_test,)
            trial_basis = self.visit(a_node if isinstance(a_node, TrialFunction) else b_node)  # 1-D (n_trial,)

            if self.ctx['is_rhs']:
                # Shouldn’t happen for dot(u,v) in a RHS, but keep it safe.
                return test_basis * trial_basis
            else:
                # Correct n_test × n_trial outer product for the local element matrix
                return np.outer(test_basis, trial_basis)

        # -----------------------------------------------------------------
        # 1. vector-basis  ⋅  vector-basis  (mass matrix blocks)
        # -----------------------------------------------------------------
        if isinstance(a_node, (VectorTrialFunction, VectorTestFunction)) \
        and isinstance(b_node, (VectorTrialFunction, VectorTestFunction)):
            trial_vec = a_val if isinstance(a_node, VectorTrialFunction) else b_val
            test_vec  = b_val if isinstance(a_node, VectorTrialFunction) else a_val
            ncomp = len((a_node if isinstance(a_node, VectorTrialFunction)
                                    else b_node).space.field_names)
            return self._diag_outer(test_vec, trial_vec, ncomp)

        # -----------------------------------------------------------------
        # 2. vector-basis  ⋅  numeric-vector   or   numeric-vector ⋅ vector-basis
        #    (e.g. dot(ũ, ∇u_k[0]))
        # -----------------------------------------------------------------
        if isinstance(a_node, (VectorTrialFunction, VectorTestFunction)) \
        and (isinstance(b_val, np.ndarray) and b_val.ndim == 1):
            return self._weight_vector_basis(a_val, b_val)

        if isinstance(b_node, (VectorTrialFunction, VectorTestFunction)) \
        and (isinstance(a_val, np.ndarray) and a_val.ndim == 1):
            return self._weight_vector_basis(b_val, a_val)

        # -----------------------------------------------------------------
        # 3. Constant / VectorFunction  ⋅  Grad(scalar basis)  (already ok)
        # -----------------------------------------------------------------
        if isinstance(a_node, (VectorFunction, Constant)) and isinstance(b_node, Grad):
            vec_val = a_val                      # (2,)
            grad_op = b_val                      # (n_basis, 2) *or* (2,)
            if grad_op.ndim == 2:                # (n_basis,2) @ (2,)  → (n_basis,)
                return grad_op @ vec_val
            if grad_op.ndim == 1:                # (2,)·(2,) → scalar
                return float(np.dot(grad_op, vec_val))

        # -----------------------------------------------------------------
        # 4. purely numerical dot products (data-data) ---------------------
        # -----------------------------------------------------------------
        if isinstance(a_val, np.ndarray) and isinstance(b_val, np.ndarray):
            if a_val.ndim == 1 and b_val.ndim == 1 and a_val.shape == b_val.shape:
                return np.dot(a_val, b_val)
            if a_val.ndim == 2 and b_val.ndim == 1 and a_val.shape[1] == b_val.shape[0]:
                return a_val @ b_val
            if a_val.ndim == 1 and b_val.ndim == 2 and a_val.shape[0] == b_val.shape[0]:
                return a_val @ b_val

        # -----------------------------------------------------------------
        # anything else is still unsupported ------------------------------
        # -----------------------------------------------------------------
        raise NotImplementedError(
            f"Dot between {a_node} and {b_node} "
            "is not implemented."
        )
    

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

    def _assemble_form(self, form, matvec):
        """
        Dispatch each integral in *form* to the proper low-level routine.
        Works whether *form* is an Integral or a Form holding many integrals.
        """
        from ufl.measures import Integral

        # 1. Normalise *form* → iterable of Integrals
        if isinstance(form, Integral):
            integrals = [form]
        elif hasattr(form, "integrals"):
            integrals = form.integrals
        else:                           # e.g. Constant(0.0) on the dummy LHS
            return

        # 2. Route every integral to its dedicated assembler
        for intg in integrals:
            if not isinstance(intg, Integral):        # skip Stray constants, etc.
                continue

            kind = intg.measure.domain_type
            if   kind == "volume":          self._assemble_volume(intg, matvec)
            elif kind == "interior_facet":  self._assemble_facet(intg,  matvec)
            elif kind == "interface":       self._assemble_interface(intg, matvec)

      


    # volume ------------------------------------------------------------
    # In ufl/compilers.py

    def _assemble_volume(self, intg, matvec):
        rhs = self.ctx['is_rhs']
        # Use the first field to determine the mesh for geometry, assuming they are all on the same one.
        mesh = self.dh.fe_map[_all_fields(intg.integrand)[0]]
        q = self.qorder or mesh.poly_order + 2
        qpts, qwts = volume(mesh.element_type, q)
        terms = _split_terms(intg.integrand)
        # print(f"is_rhs: {rhs}, terms  in volume integral: {terms}")

        # Get all unique fields in the integrand to get their reference elements
        all_q_fields = _all_fields(intg.integrand)
        ref_elements = {fld: get_reference(self.dh.fe_map[fld].element_type, self.dh.fe_map[fld].poly_order)
                        for fld in all_q_fields}

        for eid, elem in enumerate(mesh.elements_list):
            self.ctx['elem_id'] = elem.id

            for coef, term in terms:
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
                    local += coef * w * detJ * val

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
    def _elm_dofs(self, func, eid):
        """Final, robust method to get all global element DOFs for any function type."""
        if func is None:
            return []
        
        # Check for any class with a `.space` attribute (VectorTrial/TestFunction)
        if hasattr(func, 'space'):
            return np.concatenate([self.dh.element_maps[f][eid] for f in func.space.field_names])

        # Check for any class with a `.field_names` attribute (VectorFunction)
        if hasattr(func, 'field_names'):
            return np.concatenate([self.dh.element_maps[f][eid] for f in func.field_names])

        # Fallback for scalar types (Function, TrialFunction, TestFunction)
        if hasattr(func, 'field_name'):
            return self.dh.element_maps[func.field_name][eid]
            
        raise TypeError(f"Cannot get element DOFs for object of type {type(func)}")
