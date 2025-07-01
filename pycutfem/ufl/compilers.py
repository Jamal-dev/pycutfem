# pycutfem/ufl/compilers.py
"""
A minimal‑yet‑robust volume‑form compiler geared towards the project’s
current needs (Taylor–Hood Stokes, Poisson, linear elasticity, etc.).

This version integrates the ``GradOpInfo`` and ``VecOpInfo`` bookkeeping and a
shape- and role-aware algebraic layer. This distinguishes between symbolic
basis functions (for unknowns) and evaluated numerical data (for knowns),
enabling robust assembly of LHS matrices and RHS vectors.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Any, Tuple, Iterable, Mapping, Union
import logging
from dataclasses import dataclass

# -------------------------------------------------------------------------
#  Project imports – only the really needed bits
# -------------------------------------------------------------------------
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.fem import transform
from pycutfem.integration import volume
from pycutfem.integration.quadrature import line_quadrature

# Symbolic building blocks
from pycutfem.ufl.expressions import (
    Constant,    TestFunction,   TrialFunction,
    VectorTestFunction, VectorTrialFunction,
    Function,    VectorFunction,
    Grad, DivOperation, Inner, Dot,
    Sum, Sub, Prod, Pos, Neg,Div, Jump, FacetNormal,
    ElementWiseConstant
)
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.measures import Integral
from pycutfem.ufl.quadrature import PolynomialDegreeEstimator
from pycutfem.ufl.helpers import VecOpInfo, GradOpInfo
from pycutfem.ufl.analytic import Analytic

logger = logging.getLogger(__name__)
_INTERFACE_TOL = 1.0e-12 # New


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

# New: Helper to identify trial and test functions in an expression
def _trial_test(expr): # New
    """Finds the first trial and test function in an expression tree.""" # New
    trial = expr.find_first(lambda n: isinstance(n, (TrialFunction, VectorTrialFunction))) # New
    test = expr.find_first(lambda n: isinstance(n, (TestFunction, VectorTestFunction))) # New
    return trial, test # New

# ========================================================================
#  The Compiler
# ========================================================================
class FormCompiler:
    """A single‑file *volume* compiler for mixed continuous Galerkin forms."""

    def __init__(self, dh: DofHandler, quadrature_order: int | None = None, assembler_hooks: Dict[str, Any] = None):
        if dh.mixed_element is None:
            raise RuntimeError("A MixedElement‑backed DofHandler is required.")
        self.dh, self.me = dh, dh.mixed_element
        self.qorder = quadrature_order
        self.ctx: Dict[str, Any] = {"hooks": assembler_hooks or {}}
        self._basis_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self.degree_estimator = PolynomialDegreeEstimator(dh)
        self._dispatch = {
            Constant: self._visit_Constant, 
            TestFunction: self._visit_TestFunction,
            TrialFunction: self._visit_TrialFunction, 
            VectorTestFunction: self._visit_VectorTestFunction,
            VectorTrialFunction: self._visit_VectorTrialFunction, 
            Function: self._visit_Function,
            VectorFunction: self._visit_VectorFunction, Grad: self._visit_Grad,
            DivOperation: self._visit_DivOperation, Sum: self._visit_Sum,
            Sub: self._visit_Sub, Prod: self._visit_Prod, Dot: self._visit_Dot,
            Inner: self._visit_Inner, Pos: self._visit_Pos, Neg: self._visit_Neg,
            Div: self._visit_Div,
            Analytic: self._visit_Analytic,
            FacetNormal: self._visit_FacetNormal,
            ElementWiseConstant: self._visit_EWC,
            Jump: self._visit_Jump,
        }

    # ============================ PUBLIC API ===============================
    def assemble(self, eq: Equation, bcs: Union[Mapping, Iterable, None] = None):
        ndofs = self.dh.total_dofs
        K = sp.lil_matrix((ndofs, ndofs))
        F = np.zeros(ndofs)
        logger.info("Assembling LHS matrix K...")
        self.ctx["rhs"] = False
        self._assemble_form(eq.a, K)
        logger.info("Assembling RHS vector F...")
        self.ctx["rhs"] = True
        self._assemble_form(eq.L, F)
        logger.info("Applying Dirichlet boundary conditions...")
        self._apply_bcs(K, F, bcs)
        logger.info("Assembly complete.")
        return K.tocsr(), F

    # ===================== VISITORS: LEAF NODES =======================
    def _b(self, fld): return self._basis_cache[fld]["val"]  # (n_loc,)
    def _g(self, fld): return self._basis_cache[fld]["grad"] # (n_loc, d)
    def _local_dofs(self): return self.dh.get_elemental_dofs(self.ctx["eid"])

    # --- Knowns (evaluated to numerical values) ---
    def _visit_Constant(self, n: Constant):
        if n.dim ==0:
            return float(n.value)
        return np.asarray(n.value)
    
    def _visit_FacetNormal(self, n: FacetNormal): # New
        """Returns the normal vector from the context.""" # New
        if 'normal' not in self.ctx: # New
            raise RuntimeError("FacetNormal accessed outside of a facet or interface integral context.") # New
        return self.ctx['normal'] # New
    
    # -- Element-wise constants ----------------------------------------
    def _visit_EWC(self, n:ElementWiseConstant):
        eid = self.ctx.get("eid")
        if eid is None:
            raise RuntimeError("ElementWiseConstant evaluated outside an element loop.")
        return n.value_on_element(eid)
    
    
    def _visit_Pos(self, n: Pos): # New
        """Evaluates operand only if on the positive side of an interface.""" # New
        # The '+' side is where phi >= 0 (a closed set)
        if 'phi_val' in self.ctx and self.ctx['phi_val'] < -_INTERFACE_TOL: # New
            # We are on the strictly negative side, so return zero.
            op_val = self._visit(n.operand) # New
            return op_val * 0.0 # Scales scalars, arrays, and Info objects to zero # New
        return self._visit(n.operand) # New

    def _visit_Neg(self, n: Neg): # New
        """Evaluates operand only if on the negative side of an interface.""" # New
        # The '-' side is where phi < 0 (an open set)
        if 'phi_val' in self.ctx and self.ctx['phi_val'] >= _INTERFACE_TOL: # New
             # We are on the positive or zero side, so return zero.
            op_val = self._visit(n.operand) # New
            return op_val * 0.0 # New
        return self._visit(n.operand) # New

    def _visit_Jump(self, n: Jump): # New
        """Robustly evaluates jump(u) = u(+) - u(-) across an interface.""" # New
        phi_orig = self.ctx.get('phi_val') # New
        # --- Evaluate positive side trace u(+) ---
        self.ctx['phi_val'] = 1.0 # Force context to be on the '+' side # New
        u_pos = self._visit(n.u_pos) # This will call _visit_Pos # New
        # --- Evaluate negative side trace u(-) ---
        self.ctx['phi_val'] = -1.0 # Force context to be on the '-' side # New
        u_neg = self._visit(n.u_neg) # This will call _visit_Neg # New
        # --- Restore context and return difference ---
        self.ctx['phi_val'] = phi_orig # New
        return u_pos - u_neg # New
###########################################################################################3
########### --- Functions and VectorFunctions (evaluated to numerical values) ---
    def _visit_Function(self, n: Function):
        logger.debug(f"Visiting Function: {n.field_name}")
        u_loc = n.get_nodal_values(self._local_dofs())      # (22,) padded
        data = [u_loc * self._b(n.field_name)]                 # one block
        return VecOpInfo(np.stack(data), role="function")

    def _visit_VectorFunction(self, n: VectorFunction):
        """VectorFunction → VecOpInfo(k, n_loc) where n_loc = 22 for Q2–Q2–Q1."""
        logger.debug(f"Visiting VectorFunction: {n.field_names}")
        u_loc = n.get_nodal_values(self._local_dofs())      # (22,) padded
        data = [u_loc * self._b(fld) for fld in n.field_names]
        return VecOpInfo(np.stack(data), role="function")

    # --- Unknowns (represented by basis functions) ---
    def _visit_TestFunction(self, n):
        logger.debug(f"Visiting TestFunction: {n.field_name}")
        return VecOpInfo(np.stack([self._b(n.field_name)]), role="test")
    def _visit_TrialFunction(self, n):
        logger.debug(f"Visiting TrialFunction: {n.field_name}")
        return VecOpInfo(np.stack([self._b(n.field_name)]), role="trial")

    def _visit_VectorTestFunction(self, n):
        logger.debug(f"Visiting VectorTestFunction: {n.field_names}")
        return VecOpInfo(np.stack([self._b(f) for f in n.field_names]), role="test")
    def _visit_VectorTrialFunction(self, n):
        logger.debug(f"Visiting VectorTrialFunction: {n.field_names}")
        return VecOpInfo(np.stack([self._b(f) for f in n.field_names]), role="trial")
    # ================== VISITORS: OPERATORS ========================
    def _visit_Grad(self, n: Grad):
        """Return GradOpInfo with shape (k, n_dofs_local, d)."""
        op = n.operand
        logger.debug(f"Entering _visit_Grad for operand type {type(op)}")

        # ------------------------------------------------------------
        # 1.  Determine role  (affects assembly later on)
        # ------------------------------------------------------------
        if isinstance(op, (TestFunction, VectorTestFunction)):
            role = "test"
        elif isinstance(op, (TrialFunction, VectorTrialFunction)):
            role = "trial"
        elif isinstance(op, (Function, VectorFunction)):
            role = "function"
        else:
            raise NotImplementedError(f"grad() not implemented for {type(op)}")

        # ------------------------------------------------------------
        # 2.  Build one gradient block per component / field
        # ------------------------------------------------------------

        if hasattr(op, "field_names"):
            fields = op.field_names
        else:
            fields = [op.field_name]

        k_blocks = []
        for fld in fields:
            g = self._g(fld)                                    # (22,2)

            if role == "function":                              # data → scale rows
                coeffs = op.get_nodal_values(self._local_dofs()) # (22,)
                g = coeffs[:, None] * g                         # (22,2)

            # For test/trial the raw g is already correct
            k_blocks.append(g)

        return GradOpInfo(np.stack(k_blocks), role=role)

    def _visit_DivOperation(self, n: DivOperation):
        grad_op = self._visit(Grad(n.operand))           # (k, n_loc, d)
        logger.debug(f"Visiting DivOperation for operand of type {type(n.operand)}, grad_op shape: {grad_op.data.shape}")

        # ∇·v  =  Σ_i ∂v_i/∂x_i   → length n_loc (22) vector
        div_vec = np.sum([grad_op.data[i, :, i]          # pick diagonal components
                        for i in range(grad_op.data.shape[0])],
                        axis=0)

        # Decide which side of the bilinear form this lives on
        op = n.operand
        if isinstance(op, (TestFunction, VectorTestFunction)):
            role = "test"
        elif isinstance(op, (TrialFunction, VectorTrialFunction)):
            role = "trial"
        elif isinstance(op, (Function, VectorFunction)):
            role = "function"
        else:
            role = "none"

        # Wrap as VecOpInfo with a single component so that _visit_Prod
        # can use the role information to orient the outer product.
        return VecOpInfo(np.stack([div_vec]), role=role)
    


    # ================= VISITORS: ALGEBRAIC ==========================
    def _visit_Sum(self, n): return self._visit(n.a) + self._visit(n.b)
    def _visit_Sub(self, n):
        a = self._visit(n.a)
        b = self._visit(n.b)
        return a - b

    def _visit_Div(self, n):
        """Handles scalar, vector, or tensor division by a scalar."""
        numerator = self._visit(n.a)
        denominator = self._visit(n.b)

        # NumPy correctly handles element-wise division of an array by a scalar.
        return numerator / denominator

    def _visit_Prod(self, n: Prod):
        a = self._visit(n.a)
        b = self._visit(n.b)
        a_data = a.data if isinstance(a, (VecOpInfo, GradOpInfo)) else a
        b_data = b.data if isinstance(b, (VecOpInfo, GradOpInfo)) else b
        a_vec = np.squeeze(a_data) 
        b_vec = np.squeeze(b_data)
        role_a = getattr(a, 'role', None)
        role_b = getattr(b, 'role', None)
        logger.debug(f"Entering _visit_Prod for  ('{n.a!r}' * '{n.b!r}') on {'RHS' if self.ctx['rhs'] else 'LHS'}") #, a.info={getattr(a, 'info', None)}, b.info={getattr(b, 'info', None)}
        # logger.debug(f"  a: {type(a)} (role={role_a}), b: {type(b)} (role={role_b})")

        # scalar * scalar multiplication
        if np.isscalar(a) and np.isscalar(b):
            return a * b
        # First, handle context-specific RHS assembly.
        if self.ctx["rhs"]:
            # Case 1: (known Function) * (Test Function) -> results in a vector
            if isinstance(a, VecOpInfo) and a.role == "function" and isinstance(b, VecOpInfo) and b.role == "test":
                if a.data.shape[0] == 1: # Scalar field * test function
                    return np.sum(a.data) * b.data[0]
                return np.einsum("kn,kn->n", a.data, b.data, optimize=True) # Vector field

            # Symmetric orientation
            if isinstance(b, VecOpInfo) and b.role == "function" and isinstance(a, VecOpInfo) and a.role == "test":
                if b.data.shape[0] == 1: # Scalar field * test function
                    return np.sum(b.data) * a.data[0]
                return np.einsum("kn,kn->n", b.data, a.data, optimize=True) # Vector field

            # Case 2: (scalar Constant) * (Test Function) -> results in a vector
            # This now takes precedence over the general scalar multiplication.
            if np.isscalar(a) and isinstance(b, VecOpInfo) and b.role == "test":
                # This correctly returns a numeric array, not a VecOpInfo object.
                return a * b.data[0]
            if np.isscalar(b) and isinstance(a, VecOpInfo) and a.role == "test":
                return b * a.data[0]
        
        # --- General Purpose Logic (used mostly for LHS) ---

        # General scalar multiplication (if not a special RHS case)
        if np.isscalar(a) and isinstance(b, (VecOpInfo, GradOpInfo, np.ndarray)):
            return a * b
        if np.isscalar(b) and isinstance(a, (VecOpInfo, GradOpInfo, np.ndarray)):
            return b * a
            
        # LHS Matrix Assembly: (Test * Trial)
        if not self.ctx["rhs"]:
            if isinstance(a, VecOpInfo) and isinstance(b, VecOpInfo):
                if a.role == "test" and b.role in {"trial", "function"}:
                    return np.einsum("kn,km->nm", a.data, b.data, optimize=True)
                if b.role == "test" and a.role in {"trial", "function"}:
                    return np.einsum("kn,km->nm", b.data, a.data, optimize=True)
        
        raise TypeError(f"Unsupported product 'type(a)={type(a)}, type(b)={type(b)}'{n.a} * {n.b}' for {'RHS' if self.ctx['rhs'] else 'LHS'}")

        

    def _visit_Dot(self, n: Dot):
        a = self._visit(n.a)
        b = self._visit(n.b)
        a_data = a.data if isinstance(a, (VecOpInfo, GradOpInfo)) else a
        b_data = b.data if isinstance(b, (VecOpInfo, GradOpInfo)) else b
        role_a = getattr(a, 'role', None)
        role_b = getattr(b, 'role', None)
        # print(f"visit dot: role_a={role_a}, role_b={role_b}, a={a}, b={b}, side: {'RHS' if self.ctx['rhs'] else 'LHS'}")
        logger.debug(f"Entering _visit_Dot for types {type(a)} . {type(b)}")

        def rhs():
            # ------------------------------------------------------------------
            # RHS  •  dot(  Function ,  Test  )   or   dot( Test , Function )
            # ------------------------------------------------------------------
            # Case 1: Function · Test  (VecOpInfo, VecOpInfo)
            if  isinstance(a, VecOpInfo) and isinstance(b, VecOpInfo):
                if a.role == "function" and b.role == "test":
                    # f_k,n · v_k,n  →  Σ_k f_k,n v_k,n   (length-n vector for RHS)
                    # return np.einsum("km,kn->n", a.data, b.data, optimize=True)
                    return a.dot_vec(b)  # function . test
                if b.role == "function" and a.role == "test":
                    # return np.einsum("km,kn->n", b.data, a.data, optimize=True)
                    return b.dot_vec(a)  # test . function


            # ------------------------------------------------------------------
            # Function · Function   (needed for  dot(grad(u_k), u_k)  on RHS)
            # ------------------------------------------------------------------
            # Case 2: Function · Function  (GradOpInfo, VecOpInfo)
            if  isinstance(a, GradOpInfo) and isinstance(b, VecOpInfo) \
            and a.role == b.role == "function":
                # u_val   = np.sum(b.data, axis=1)                # (d,)
                # data_fk = np.einsum("knd,d->kn", a.data, u_val) # keep basis-index n
                # return VecOpInfo(data_fk, role="function")
                return a.dot_vec(b)  # grad(u_k) . u_k
            if  isinstance(b, GradOpInfo) and isinstance(a, VecOpInfo) \
            and b.role == a.role == "function":
                # u_val   = np.sum(a.data, axis=1)                # (d,)
                # data_fk = np.einsum("knd,d->kn", b.data, u_val) # keep basis-index n
                # return VecOpInfo(data_fk, role="function")
                return b.dot_vec(a)
            
            
            # Constant (numpy 1-D) · test VecOpInfo  → length-n vector
            # Case 3: Constant(np.array([u1,u2])) · Test
            if isinstance(a, np.ndarray) and a.ndim == 1 and \
            isinstance(b, VecOpInfo) and b.role == "test":
                return b.dot_const(a)

            if isinstance(b, np.ndarray) and b.ndim == 1 and \
            isinstance(a, VecOpInfo) and a.role == "test":
                return a.dot_const(b)
            
            # Case 4: Constant(np.array([u1,u2])) · Trial or Function, no test so output is VecOpInfo
            if isinstance(a, np.ndarray) and a.ndim == 1 and \
             isinstance(b, (VecOpInfo, GradOpInfo)) and b.role in {"trial", "function"}:
                # return b.dot_const(a) if b.ndim==2 else b.dot_vec(a)
                return b.dot_const_vec(a) if b.ndim==2 else b.dot_vec(a)
            if isinstance(b, np.ndarray) and b.ndim == 1 and \
             isinstance(a, (VecOpInfo, GradOpInfo)) and a.role in {"trial", "function"}:
                # return a.dot_const(b) if a.ndim==2 else a.dot_vec(b)
                return a.dot_const_vec(b) if a.ndim==2 else a.dot_vec(b)

        if self.ctx["rhs"]:
            result = rhs()
            if result is not None:
                return result
            else:
                raise TypeError(f"Unsupported dot product for RHS: '{n.a} . {n.b}'")
        
        if role_a == None and role_b == "test":
            # Special case like dot( Constat(np.array([u1,u2])), TestFunction('v') )
            # This is a special case where we have a numerical vector on the LHS
            # and a test function basis on the RHS.
            if isinstance(a, np.ndarray) and isinstance(b, VecOpInfo) and b.role == "test":
                return b.dot_const(a)
            if isinstance(b, np.ndarray) and isinstance(a, VecOpInfo) and a.role == "test":
                return a.dot_const(b)
        # Dot product between a basis field and a numerical vector
        if isinstance(a, (VecOpInfo, GradOpInfo)) and isinstance(b, np.ndarray): return a.dot_const_vec(b) if a.ndim==2 else a.dot_vec(b)
        if isinstance(b, (VecOpInfo, GradOpInfo)) and isinstance(a, np.ndarray): return b.dot_const_vec(a) if b.ndim==2 else b.dot_vec(a)
        

        # mass matrix case: VecOpInfo . VecOpInfo
        if isinstance(a, VecOpInfo) and isinstance(b, VecOpInfo):
            logger.debug(f"visit dot: Both operands are VecOpInfo: {a.role} . {b.role}")
            if a.role == "test" and b.role in {"trial", "function"}:
                # return np.dot(a_data.T, b_data)  # test . trial
                return b.dot_vec(a)  # test . trial
            elif b.role == "test" and a.role in {"trial", "function"}:
                # return np.dot(b_data.T, a_data)  # tiral . test
                return a.dot_vec(b)  # trial . test
        
        
        # ------------------------------------------------------------------
        # case grad(u_trial) . u_k
        if isinstance(a, GradOpInfo) and ((isinstance(b, VecOpInfo) \
            and (b.role == "function" )) and a.role == "trial" 
            ):

            # velocity value at this quadrature point
            # u_val = np.sum(b.data, axis=1)                 # (d,)

            # works for k = 1 (scalar) and k = 2 (vector) alike
            # data  = np.einsum("knd,d->kn", a.data, u_val, optimize=True)

            # return VecOpInfo(data, role=a.role)
            return a.dot_vec(b)  # grad(u) . u_k
        
        # ------------------------------------------------------------------
        # Case:  VectorFunction · Grad(⋅)      u_k · ∇w_test
        # ------------------------------------------------------------------
        if isinstance(a, VecOpInfo) and isinstance(b, GradOpInfo) \
        and a.role == "function" and b.role == "test":

            # 1. velocity value at the current quadrature point
            # u_val = np.sum(a.data, axis=1)                  # (d,)  —   u_d(ξ)

            # # 2. dot with each gradient row   w_{k,n} = Σ_d u_d ∂_d φ_{k,n}
            # #    Works for both scalar (k = 1) and vector (k = 2) targets.
            # data  = np.einsum("d,knd->kn", u_val, b.data, optimize=True)

            # 3. The role is inherited from the Grad operand (trial / test / function)
            # return VecOpInfo(data, role=b.role)
            return a.dot_grad(b)  # u_k · ∇w
        
        # ------------------------------------------------------------------
        # Case:  Grad(Function) · Vec(Trial)      (∇u_k) · u
        # ------------------------------------------------------------------
        if isinstance(a, GradOpInfo) and a.role == "function" \
        and isinstance(b, VecOpInfo)  and b.role == "trial":

            # (1)  value of ∇u_k at this quad-point
            # grad_val = np.sum(a.data, axis=1)          # (k,d)

            # # (2)  w_i,n = Σ_d (∇u_k)_i,d  *  φ_{u,d,n}
            # data = np.einsum("kd,kn->kn", grad_val, b.data, optimize=True)

            # return VecOpInfo(data, role="trial")
            return a.dot_func(b)
        
        # ------------------------------------------------------------------
        # Case:  Vec(Function) · Grad(Trial)      u_k · ∇u_trial
        # ------------------------------------------------------------------
        if isinstance(a, VecOpInfo)  and a.role == "function" \
        and isinstance(b, GradOpInfo) and b.role == "trial":

            return a.dot_grad(b)  # u_k · ∇u_trial
            # # print(f"problem " * 4)
            # # return a.dot_grad(b)  # u_k · ∇u
            # # trying new things here
            # fun_vals = np.sum(a.data, axis=1) 
            # fun_vals_x = sum(a.data[0,:])
            # fun_vals_y = sum(a.data[1,:])
            # # print(f"func_vals_x: {fun_vals_x}, sum(fun_vals_x): {np.sum(fun_vals_x)}") 
            # # print(f"func_vals_y: {fun_vals_y}, sum(fun_vals_y): {np.sum(fun_vals_y)}")
            # if b.data.shape[0] == 2: # vector gradient
            #     a11 = b.data[0,:,0] # ∂_x u_trial_1
            #     a12 = b.data[0,:,1] # ∂_y u_trial_1
            #     a21 = b.data[1,:,0] # ∂_x u_trial_2
            #     a22 = b.data[1,:,1] # ∂_y u_trial_2
            #     c11 =  fun_vals_x * a11 +  fun_vals_y * a21 # ∂_x u_trial
            #     c12 =  fun_vals_x * a12 +  fun_vals_y * a22 #
            #     # print(f"shapes = c11.shape:{c11.shape}, c12.shape:{c12.shape}, {fun_vals_x.shape}, fun_vals_y.shape:{fun_vals_y.shape}, a11.shape:{a11.shape}, a12.shape:{a12.shape}, a21.shape:{a21.shape}, a22.shape:{a22.shape}")
            #     c = np.stack([c11, c12], axis=0) # (2, n_loc)
            # else: # scalar gradient
            #     a1 = b.data[0,:,0] # ∂_x u_trial
            #     a2 = b.data[0,:,1] # ∂_y u_trial
            #     c1 = fun_vals_x *  a1 + fun_vals_y *  a2 # ∂_x u_trial
            #     c = np.stack([c1], axis=0)
            # # data = np.einsum("km,imk->im", a.data, b.data, optimize=True)
            # return VecOpInfo(c, role="trial")  # u_k · ∇u_trial
        
        # Both are numerical vectors (RHS)
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray): return np.dot(a,b)

        raise TypeError(f"Unsupported dot product '{n.a} . {n.b}'")

    def _visit_Inner(self, n: Inner):
        a = self._visit(n.a)
        b = self._visit(n.b)
        logger.debug(f"Entering _visit_Inner for types {type(a)} : {type(b)}")

        # ------------------------------------------------------------------
        # RHS:  inner( Grad(Function) , Grad(Test) )   or   symmetric case
        #        → length-n vector  F_n = Σ_{k,d}  (∑_m ∂_d u_k φ_m) · ∂_d v_k,n
        # ------------------------------------------------------------------
        if self.ctx.get("rhs"):
            if isinstance(a, GradOpInfo) and isinstance(b, GradOpInfo):
                # Function · Test  ............................................
                if a.role == "function" and b.role == "test":
                    grad_val = np.sum(a.data, axis=1)            # (k,d)  ∇u_k(x_q)
                    return np.einsum("kd,knd->n", grad_val, b.data, optimize=True)

                # Test · Function  (rare but symmetrical) .............
                if b.role == "function" and a.role == "test":
                    grad_val = np.sum(b.data, axis=1)            # (k,d)
                    return np.einsum("kd,knd->n", grad_val, a.data, optimize=True)

        # Both are Info objects of the same kind -> matrix assembly
        if type(a) is type(b) and isinstance(a, (VecOpInfo, GradOpInfo)):
            # print(f"k"*32)
            return a.inner(b)

        # One is a basis, the other a numerical tensor value (RHS or complex LHS)
        if isinstance(a, GradOpInfo) and isinstance(b, np.ndarray): 
            # print(f"*"*32)
            return a.contracted_with_tensor(b)
        if isinstance(b, GradOpInfo) and isinstance(a, np.ndarray): 
            # print(f"@"*32)
            return b.contracted_with_tensor(a)
        
        # RHS: both are functions, result is scalar integral
        if self.ctx['rhs'] and isinstance(a, VecOpInfo) and isinstance(b, VecOpInfo):
            #  print(f"&"*32)
             return np.sum(a.data * b.data)

        raise TypeError(f"Unsupported inner product '{n.a} : {n.b}'")
        
    # Visitor dispatch
    def _visit(self, node): return self._dispatch[type(node)](node)

    def _visit_Analytic(self, node): return node.eval(self.ctx['x_phys'])

    # ======================= ASSEMBLY CORE ============================
    def _assemble_form(self, form, target): # NEW
        """Accept a Form object and iterate through its integrals.""" # NEW
        from pycutfem.ufl.forms import Form
        if form is None or not isinstance(form, Form): return # NEW

        for integral in form.integrals: # NEW
            if not isinstance(integral, Integral): continue # NEW
            if integral.measure.domain_type == "interface":
                # Handle interface integrals with a level set
                logger.info(f"Assembling interface integral: {integral}")
                self._assemble_interface(integral, target)
                continue
            if integral.measure.domain_type == "volume":
                # Handle volume integrals
                logger.info(f"Assembling volume integral: {integral}")
                self._assemble_volume(integral, target)
                continue
            if integral.measure.domain_type not in ["volume", "interface"]:
                logger.warning(f"Skipping unsupported integral type: {integral.measure.domain_type}")
                raise NotImplementedError(f"Unsupported integral type: {integral.measure.domain_type}")

    def _find_q_order(self, integral: Integral) -> int:
        # --- 1. explicit override in the dx metadata -------------------------
        q_order = (integral.measure.metadata.get('quad_degree')     #   Fenics
                or integral.measure.metadata.get('quad_order')   # ← new alias
                or integral.measure.metadata.get('q'))           #   VTK / misc.

        # --- 2. global ‘assemble_form(…, quad_order=…)’ fallback -------------
        if q_order is None and self.qorder is not None:
            q_order = self.qorder

        # --- 3. last resort: automatic estimation ----------------------------
        if q_order is None:                       # only if nothing fixed it yet
            try:
                p = self.degree_estimator.estimate_degree(integral.integrand)
                q_order = max(1, math.ceil((p + 1) / 2))            # k ≥ (p+1)/2
                logger.debug(f'Auto-detected quad_order={q_order} for integral.')
            except Exception as e:
                logger.warning(f'Could not estimate polynomial degree: {e}.')
                q_order = max(1, self.me.mesh.poly_order * 2)

        return q_order
    
    # ====================== BC handling ===============================
    def _apply_bcs(self, K, F, bcs):
        # This method remains unchanged from your provided file
        if not bcs: return
        data = self.dh.get_dirichlet_data(bcs)
        if not data: return
        rows = np.fromiter(data.keys(), dtype=int)
        vals = np.fromiter(data.values(), dtype=float)
        
        # Apply to RHS vector F
        F -= K @ np.bincount(rows, weights=vals, minlength=F.size)
        
        # Zero out rows and columns in the matrix
        K_lil = K.tolil()
        K_lil[rows, :] = 0
        K_lil[:, rows] = 0
        K_lil[rows, rows] = 1.0
        # copy the edited data back
        K[:] = K_lil  # Convert back to CSR format
        
        # Set values in the RHS vector
        F[rows] = vals
    
    def _assemble_volume(self, integral: Integral, matvec):
        mesh = self.me.mesh
        # Use a higher quadrature order for safety, esp. with mixed orders
        q_order = self._find_q_order(integral)
        qp, qw = volume(mesh.element_type, q_order)
        fields = _all_fields(integral.integrand)
        rhs = self.ctx["rhs"]

        for eid, element in enumerate(mesh.elements_list):
            if rhs:
                loc = np.zeros(self.me.n_dofs_local)
            else:
                loc = np.zeros((self.me.n_dofs_local, self.me.n_dofs_local))
            
            for (xi, eta), w in zip(qp, qw):
                J = transform.jacobian(mesh, eid, (xi, eta))
                detJ = abs(np.linalg.det(J))
                JiT = np.linalg.inv(J).T
                self.ctx['x_phys'] = transform.x_mapping(mesh, eid, (xi, eta))
                
                # Cache basis values and gradients for this quadrature point
                self._basis_cache.clear()
                for f in fields:
                    val = self.me.basis(f, xi, eta)
                    g_ref = self.me.grad_basis(f, xi, eta)
                    self._basis_cache[f] = {"val": val, "grad": g_ref @ JiT}
                
                self.ctx["eid"] = eid
                integrand_val = self._visit(integral.integrand)
                loc += w * detJ * integrand_val
            
            gdofs = self.dh.get_elemental_dofs(eid)
            if rhs:
                np.add.at(matvec, gdofs, loc)
            else:
                # Efficiently add local matrix to sparse global matrix
                r, c = np.meshgrid(gdofs, gdofs, indexing="ij")
                matvec[r, c] += loc
        
        self.ctx.pop("eid", None); self.ctx.pop("x_phys", None)  # Clean up context

    
    
    def _assemble_interface(self, intg: Integral, matvec): # New
        """ # New
        Assembles integrals over non-conforming interfaces defined by a level set. # New
        This implementation is adapted for the new MixedElement/DofHandler architecture. # New
        """ # New
        log = logging.getLogger(__name__) # New
        log.debug(f"Assembling interface integral: {intg}") # New

        # --- 1. Initial Setup --- # New
        rhs = self.ctx['rhs'] # New
        level_set = intg.measure.level_set # New
        if level_set is None: # New
            raise ValueError("dInterface measure requires a level_set.") # New
            
        mesh = self.me.mesh # New
        qdeg = self._find_q_order(intg) 
        logger.debug(f"Assemble Interface: Using quadrature degree: {qdeg}") 
        fields = _all_fields(intg.integrand) # New
        
        hook = self.ctx['hooks'].get(type(intg.integrand)) # New
        if hook: # New
            self.ctx.setdefault('scalar_results', {})[hook['name']] = 0.0 # New

        try: # New
            # --- 2. Loop Over All Cut Elements in the Mesh --- # New
            for elem in mesh.elements_list: # New
                if elem.tag != 'cut' or len(elem.interface_pts) != 2: # New
                    continue # New
                
                log.debug(f"  Processing cut element: {elem.id}") # New

                # --- 3. Quadrature Rule on the Physical Interface Segment --- # New
                p0, p1 = elem.interface_pts # New
                qp, qw = line_quadrature(p0, p1, qdeg) # New
                
                # --- 4. Determine Assembly Path (Bilinear, Linear, or Functional) --- # New
                trial, test = _trial_test(intg.integrand) # New

                # --- PATH A: Bilinear Form (e.g., a(u,v), contributes to the matrix) --- # New
                if not rhs and trial and test: # New
                    loc = np.zeros((self.me.n_dofs_local, self.me.n_dofs_local)) # New
                    
                    for x_phys, w in zip(qp, qw): # New
                        # --- Set context for the current quadrature point --- # New
                        xi, eta = transform.inverse_mapping(mesh, elem.id, x_phys) # New
                        J = transform.jacobian(mesh, elem.id, (xi, eta)) # New
                        JiT = np.linalg.inv(J).T # New
                        
                        self._basis_cache.clear() # New
                        for f in fields: # New
                            val = self.me.basis(f, xi, eta) # New
                            grad = self.me.grad_basis(f, xi, eta) @ JiT # New
                            self._basis_cache[f] = {"val": val, "grad": grad} # New

                        self.ctx['eid'] = elem.id # New
                        self.ctx['normal'] = level_set.gradient(x_phys) # New
                        self.ctx['phi_val'] = level_set(x_phys) # New
                        self.ctx['x_phys'] = x_phys
                        
                        # --- Evaluate the ENTIRE integrand at once --- # New
                        integrand_val = self._visit(intg.integrand) # New
                        loc += w * integrand_val # w contains the 1D jacobian # New
                    
                    gdofs = self.dh.get_elemental_dofs(elem.id) # New
                    r, c = np.meshgrid(gdofs, gdofs, indexing="ij") # New
                    matvec[r, c] += loc # New
                    log.debug(f"    Assembled {loc.shape} local matrix for element {elem.id}") # New

                # --- PATH B: Linear Form (e.g., L(v)) or Functional (e.g., jump(u)) --- # New
                else: # New
                    acc = None # New
                    for x_phys, w in zip(qp, qw): # New
                        xi, eta = transform.inverse_mapping(mesh, elem.id, x_phys) # New
                        J = transform.jacobian(mesh, elem.id, (xi, eta)) # New
                        JiT = np.linalg.inv(J).T # New

                        self._basis_cache.clear() # New
                        for f in fields: # New
                            val = self.me.basis(f, xi, eta) # New
                            grad = self.me.grad_basis(f, xi, eta) @ JiT # New
                            self._basis_cache[f] = {"val": val, "grad": grad} # New
                        
                        self.ctx['eid'] = elem.id # New
                        self.ctx['normal'] = level_set.gradient(x_phys) # New
                        self.ctx['phi_val'] = level_set(x_phys) # New
                        self.ctx['x_phys'] = x_phys
                        
                        integrand_val = self._visit(intg.integrand) # New
                        
                        if acc is None: # New
                            acc = w * np.asarray(integrand_val) # New
                        else: # New
                            acc += w * np.asarray(integrand_val) # New

                    if acc is not None: # New
                        if rhs and test: # It's a linear form # New
                            gdofs = self.dh.get_elemental_dofs(elem.id) # New
                            np.add.at(matvec, gdofs, acc) # New
                            log.debug(f"    Assembled {acc.shape} local vector for element {elem.id}") # New
                        elif hook: # It's a hooked functional 
                            if isinstance(acc,VecOpInfo): # New
                                acc = np.sum(acc.data, axis=1)
                            self.ctx['scalar_results'][hook['name']] += acc
                            log.debug(f"    Accumulated functional '{hook['name']}' for element {elem.id}") # New

        finally: # New
            # --- Final Context Cleanup --- # New
            for key in ('phi_val', 'normal', 'eid', 'x_phys'): # New
                self.ctx.pop(key, None) # New
            log.debug("Interface assembly finished. Context cleaned.") # New