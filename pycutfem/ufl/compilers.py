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
import math
import os

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
    ElementWiseConstant, Derivative
)
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.measures import Integral
from pycutfem.ufl.quadrature import PolynomialDegreeEstimator
from pycutfem.ufl.helpers import VecOpInfo, GradOpInfo, required_multi_indices
from pycutfem.fem.transform import map_deriv
from pycutfem.ufl.analytic import Analytic
from pycutfem.utils.domain_manager import get_domain_bitset
from pycutfem.ufl.helpers_jit import  _build_jit_kernel_args, _scatter_element_contribs, _stack_ragged



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

def _find_all(expr, cls):
    out = []
    def walk(n):
        if isinstance(n, cls):
            out.append(n)
        for attr in ('operand','a','b','u_pos','u_neg','components','f'):
            m = getattr(n, attr, None)
            if m is None: continue
            if isinstance(m, (list, tuple)):
                for x in m: walk(x)
            else:
                walk(m)
    walk(expr)
    return out

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

    def __init__(self, dh: DofHandler, quadrature_order: int | None = None, 
                 assembler_hooks: Dict[str, Any] = None,
                 backend: str = "jit"):
        if dh.mixed_element is None:
            raise RuntimeError("A MixedElement‑backed DofHandler is required.")
        self.dh, self.me = dh, dh.mixed_element
        self.qorder = quadrature_order
        self.ctx: Dict[str, Any] = {"hooks": assembler_hooks or {}}
        self._basis_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self.degree_estimator = PolynomialDegreeEstimator(dh)
        self.backend = backend
        if self.backend == "jit":
            from pycutfem.jit import compile_backend
            self._compile_backend = compile_backend
        self._dispatch = {
            Constant: self._visit_Constant, 
            TestFunction: self._visit_TestFunction,
            TrialFunction: self._visit_TrialFunction, 
            VectorTestFunction: self._visit_VectorTestFunction,
            VectorTrialFunction: self._visit_VectorTrialFunction, 
            Function: self._visit_Function,
            VectorFunction: self._visit_VectorFunction, 
            Grad: self._visit_Grad,
            DivOperation: self._visit_DivOperation, Sum: self._visit_Sum,
            Sub: self._visit_Sub, Prod: self._visit_Prod, Dot: self._visit_Dot,
            Inner: self._visit_Inner, Pos: self._visit_Pos, Neg: self._visit_Neg,
            Div: self._visit_Div,
            Analytic: self._visit_Analytic,
            FacetNormal: self._visit_FacetNormal,
            ElementWiseConstant: self._visit_EWC,
            Jump: self._visit_Jump,
            Derivative        : self._visit_Derivative
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
    def _b(self, fld): 
        # ➊ standard volume/interface path
        if fld in self._basis_cache:
            return self._basis_cache[fld]["val"]

        # ➋ ghost-edge fallback: pull from the side-aware cache
        bv = self.ctx.get("basis_values")
        if bv is not None:
            side = '+' if self.ctx.get("phi_val", 0.0) >= 0 else '-'
            if fld in bv[side] and (0, 0) in bv[side][fld]:
                return bv[side][fld][(0, 0)]
        raise KeyError(f"Basis for field '{fld}' not found.")
    def _g(self, fld):
        if fld in self._basis_cache:
            return self._basis_cache[fld]["grad"]

        bv = self.ctx.get("basis_values")
        if bv is not None:
            side = '+' if self.ctx.get("phi_val", 0.0) >= 0 else '-'
            gx = bv[side][fld].get((1, 0))
            gy = bv[side][fld].get((0, 1))
            if gx is not None and gy is not None:
                return np.stack([gx, gy], axis=1)
        raise KeyError(f"Gradient for field '{fld}' not found.")
    def _lookup_basis(self, field: str, alpha: tuple[int, int]) -> np.ndarray:
        """
        Return the trace of ∂^{alpha}φ_i (i = 1..n_loc) at the current
        quadrature point.

        * If we are on a ghost edge   →  use the side–aware cache.
        * Otherwise (volume / interior face) fall back to the element cache
        exposed via _b / _g.
        """
        # ---- ghost-edge path ------------------------------------------
        if "basis_values" in self.ctx:
            side = '+' if self.ctx.get("phi_val", 0.0) >= 0 else '-'
            return self.ctx["basis_values"][side][field][alpha]

        # ---- standard element path ------------------------------------
        if alpha == (0, 0):
            return self._b(field)                                             # (n_loc,)
        if alpha == (1, 0):
            return self._g(field)[:, 0]                                       # (n_loc,)
        if alpha == (0, 1):
            return self._g(field)[:, 1]                                       # (n_loc,)

        # Higher derivatives outside ghost-edge assembly are not (yet) required
        raise KeyError(f"Basis for derivative {alpha} of field '{field}' not found.")
    def _local_dofs(self): 
        if 'eid' in self.ctx:           # matrix/vector path
            return self.dh.get_elemental_dofs(self.ctx["eid"])
        return self.ctx['global_dofs']   # functional path

    def _visit_Derivative(self, op: Derivative):
        # --------------------------------------------------------------
        # 1) Operand is a Jump  →  Jump of derivatives
        # --------------------------------------------------------------
        if isinstance(op.f, Jump):
            # Build two new Derivative nodes with the same multi-index
            dpos = Derivative(op.f.u_pos, *op.order)
            dneg = Derivative(op.f.u_neg, *op.order)

            phi_old = self.ctx.get('phi_val', 0.0)
            eid_old = self.ctx.get('eid')

            # + side
            self.ctx['phi_val'] = 1.0
            if 'pos_eid' in self.ctx:
                self.ctx['eid'] = self.ctx['pos_eid']
            v_pos = self._visit(dpos)

            # – side
            self.ctx['phi_val'] = -1.0
            if 'neg_eid' in self.ctx:
                self.ctx['eid'] = self.ctx['neg_eid']
            v_neg = self._visit(dneg)

            # restore
            self.ctx['phi_val'] = phi_old
            if eid_old is None:
                self.ctx.pop('eid', None)
            else:
                self.ctx['eid'] = eid_old

            return v_pos - v_neg

        # --------------------------------------------------------------
        # 2) Operand is *not* a Jump (previous logic, unchanged)
        # --------------------------------------------------------------
        fld, alpha = op.f.field_name, op.order
        row = self._lookup_basis(fld, alpha)[np.newaxis, :]      # (1,n)

        if op.f.is_function:
            coeffs = op.f.get_nodal_values(self._local_dofs())[np.newaxis, :]
            data, role = coeffs * row, "function"
        elif op.f.is_trial:
            data, role = row, "trial"
        else:
            data, role = row, "test"

        return VecOpInfo(data, role=role)
   
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
    
    
    def _visit_Pos(self, n: Pos): 
        """Evaluates operand only if on the positive side of an interface.""" 
        # The '+' side is where phi >= 0 (a closed set)
        if 'phi_val' in self.ctx and self.ctx['phi_val'] < -_INTERFACE_TOL: 
            # We are on the strictly negative side, so return zero.
            op_val = self._visit(n.operand) 
            return op_val * 0.0 # Scales scalars, arrays, and Info objects to zero 
        return self._visit(n.operand) 

    def _visit_Neg(self, n: Neg): 
        """Evaluates operand only if on the negative side of an interface.""" 
        # The '-' side is where phi < 0 (an open set)
        if 'phi_val' in self.ctx and self.ctx['phi_val'] >= _INTERFACE_TOL: 
             # We are on the positive or zero side, so return zero.
            op_val = self._visit(n.operand) 
            return op_val * 0.0 
        return self._visit(n.operand) 

    def _visit_Jump(self, n: Jump): 
        """Robustly evaluates jump(u) = u(+) - u(-) across an interface.""" 
        phi_orig = self.ctx.get('phi_val') 
        eid_orig  = self.ctx.get('eid')
        # --- Evaluate positive side trace u(+) ---
        self.ctx['eid']     = self.ctx.get('pos_eid')
        self.ctx['phi_val'] = 1.0 # Force context to be on the '+' side 
        u_pos = self._visit(n.u_pos) # This will call _visit_Pos 
        # --- Evaluate negative side trace u(-) ---
        self.ctx['eid']     = self.ctx.get('neg_eid')
        self.ctx['phi_val'] = -1.0 # Force context to be on the '-' side 
        u_neg = self._visit(n.u_neg) # This will call _visit_Neg 
        # --- Restore context and return difference ---
        self.ctx['phi_val'] = phi_orig 
        if eid_orig is not None:
            self.ctx['eid'] = eid_orig
        else:
            self.ctx.pop('eid', None)
        return u_pos - u_neg 
###########################################################################################3
########### --- Functions and VectorFunctions (evaluated to numerical values) ---
    def _visit_Function(self, n: Function):
        logger.debug(f"Visiting Function: {n.field_name}")
        u_loc = n.get_nodal_values(self._local_dofs())      # element-local coeffs
        phi   = self._b(n.field_name)                       # trace on current side

        # ghost-edge: enlarge coeff vector to |global_dofs|
        if phi.shape[0] != u_loc.shape[0] and "global_dofs" in self.ctx:
            padded = np.zeros_like(phi)
            side   = '+' if self.ctx.get("phi_val", 0.0) >= 0 else '-'
            amap   = self.ctx["pos_map"] if side == '+' else self.ctx["neg_map"]
            padded[amap] = u_loc
            u_loc = padded

        data = [u_loc * phi]
        return VecOpInfo(np.stack(data), role="function")

    def _visit_VectorFunction(self, n: VectorFunction):
        """
        Return VecOpInfo(k, n_loc) with k = len(field_names).
        Handles both volume and ghost-edge assembly: coefficients are padded
        exactly as in _visit_Function when we are on a ghost edge.
        """
        logger.debug(f"Visiting VectorFunction: {n.field_names}")
        data = []
        local_dofs = self._local_dofs()                          # element or union
        for i, fld in enumerate(n.field_names):
            coeffs = n.components[i].get_nodal_values(local_dofs)  # (n_loc,)
            phi    = self._b(fld)                                   # (n_loc,) – padded on ghost edge

            # --- pad coefficients on ghost edges --------------------
            if phi.shape[0] != coeffs.shape[0] and 'global_dofs' in self.ctx:
                padded = np.zeros_like(phi)
                side   = '+' if self.ctx.get('phi_val', 0.0) >= 0 else '-'
                amap   = self.ctx['pos_map'] if side == '+' else self.ctx['neg_map']
                padded[amap] = coeffs
                coeffs = padded

            data.append(coeffs * phi)                              # component-wise
        return VecOpInfo(np.stack(data), role="function")


    # --- Unknowns (represented by basis functions) ---
    def _visit_TestFunction(self, n):
        logger.debug(f"Visiting TestFunction: {n.field_name}")
        row = self._lookup_basis(n.field_name, (0, 0))[np.newaxis, :]
        return VecOpInfo(row, role="test")
    def _visit_TrialFunction(self, n):
        logger.debug(f"Visiting TrialFunction: {n.field_name}")
        row = self._lookup_basis(n.field_name, (0, 0))[np.newaxis, :]
        return VecOpInfo(row, role="trial")

    def _visit_VectorTestFunction(self, n):
        logger.debug(f"Visiting VectorTestFunction: {n.field_names}")
        return VecOpInfo(np.stack([self._lookup_basis(f, (0, 0)) for f in n.field_names]), role="test")
    def _visit_VectorTrialFunction(self, n):
        logger.debug(f"Visiting VectorTrialFunction: {n.field_names}")
        return VecOpInfo(np.stack([self._lookup_basis(f, (0, 0)) for f in n.field_names]), role="trial")
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
        coeffs_list = []
        for i,fld in enumerate(fields):
            
            g = self._g(fld)                                    # (22,2)

            if role == "function":                              # data → scale rows
                # coeffs = op.get_nodal_values(self._local_dofs()) # (22,)
                if hasattr(op, "components"):
                    # VectorFunction case: each component has its own values
                    coeffs = op.components[i].padded_values(self._local_dofs()) 
                    # print(f"field {i}: coeffs({coeffs.shape}) : , {coeffs}")
                else:
                    # Function case: single component
                    coeffs = op.padded_values(self._local_dofs()) # (22,)
                # g = coeffs[:, None] * g                         # (22,2)
                coeffs_list.append(coeffs)  # Store coeffs for later use
                # print(f"field {i}: g({g.shape}) : , {g}")

            # For test/trial the raw g is already correct
            k_blocks.append(g)

        if role == "function":
            # Pass the un-scaled gradients and the coefficients separately
            return GradOpInfo(np.stack(k_blocks), role=role, coeffs=np.stack(coeffs_list))
        else:
            # Trial/Test functions have no coeffs, this is unchanged
            return GradOpInfo(np.stack(k_blocks), role=role)

    def _visit_DivOperation(self, n: DivOperation):
        grad_op = self._visit(Grad(n.operand))           # (k, n_loc, d)
        logger.debug(f"Visiting DivOperation for operand of type {type(n.operand)}, grad_op shape: {grad_op.data.shape}")

        if grad_op.role == "function":
            # scaled the gradient
            grad_val = np.einsum("knd,kn->kd", grad_op.data, grad_op.coeffs, optimize=True) # (k,d) (2,2)
            div_vec = np.sum([grad_val[i, i] for i in range(grad_val.shape[0])], axis=0)  # sum over k
        else:
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
        # a_vec = np.squeeze(a_data) 
        # b_vec = np.squeeze(b_data)
        # role_a = getattr(a, 'role', None)
        # role_b = getattr(b, 'role', None)
        logger.debug(f"Entering _visit_Prod for  ('{n.a!r}' * '{n.b!r}') on {'RHS' if self.ctx['rhs'] else 'LHS'}") #, a.info={getattr(a, 'info', None)}, b.info={getattr(b, 'info', None)}
        # logger.debug(f"  a: {type(a)} (role={role_a}), b: {type(b)} (role={role_b})")

        # scalar * scalar multiplication
        if np.isscalar(a) and np.isscalar(b):
            return a * b
        # RHS: (Function) * (Function)  → scalar
        if (isinstance(a, VecOpInfo) and isinstance(b, VecOpInfo)
            and a.role == b.role == "function"):
            return a * b          # delegates to VecOpInfo.__mul_
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
                return a * b
        
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

                    return a.dot_vec(b)  # function . test
                if b.role == "function" and a.role == "test":
                    
                    return b.dot_vec(a)  # test . function


            # ------------------------------------------------------------------
            # Function · Function   (needed for  dot(grad(u_k), u_k)  on RHS)
            # ------------------------------------------------------------------
            # Case 2: Function · Function  (GradOpInfo, VecOpInfo)
            if  isinstance(a, GradOpInfo) and isinstance(b, VecOpInfo) \
            and a.role == b.role == "function":
                return a.dot_vec(b)  # grad(u_k) . u_k
            if  isinstance(b, GradOpInfo) and isinstance(a, VecOpInfo) \
            and b.role == a.role == "function":
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
        # Case:  VectorFunction · Grad(⋅)      u_k · ∇w_test
        # ------------------------------------------------------------------
        if isinstance(a, VecOpInfo) and isinstance(b, GradOpInfo) \
        and a.role == "function" and b.role == "test":

            return a.dot_grad(b)  # u_k · ∇w
        
        # ------------------------------------------------------------------
        # case grad(u_trial) . u_k
        if isinstance(a, GradOpInfo) and ((isinstance(b, VecOpInfo) \
            and (b.role == "function" )) and a.role == "trial" 
            ):

            return a.dot_vec(b)  # ∇u_trial · u_k
        # ------------------------------------------------------------------
        # case u_trial . grad(u_k)
        if isinstance(b, GradOpInfo) and ((isinstance(a, VecOpInfo) \
            and (a.role == "trial" )) and b.role == "function"
            ):

            return a.dot_grad(b)  # u_trial · ∇u_k
        
 
        
        
        # ------------------------------------------------------------------
        # Case:  Grad(Function) · Vec(Trial)      (∇u_k) · u
        # ------------------------------------------------------------------
        if isinstance(a, GradOpInfo) and a.role == "function" \
        and isinstance(b, VecOpInfo)  and b.role == "trial":

            return a.dot_vec(b)  # ∇u_k · u_trial
        
        # ------------------------------------------------------------------
        # Case:  Vec(Function) · Grad(Trial)      u_k · ∇u_trial
        # ------------------------------------------------------------------
        if isinstance(a, VecOpInfo)  and a.role == "function" \
        and isinstance(b, GradOpInfo) and b.role == "trial":

            return a.dot_grad(b)  # u_k · ∇u_trial
            
        
        # Both are numerical vectors (RHS)
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray): return np.dot(a,b)

        raise TypeError(f"Unsupported dot product '{n.a} . {n.b}'")

    
    # ================ VISITORS: INNER PRODUCTS =========================
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
                    # print(f"mei hu na" * 6)
                    # print(f"a.data before summition: {a.data}")

                    grad_val = np.einsum("knd,kn->kd", a.data, a.coeffs, optimize=True)
                    # print(f"grad_val: {grad_val}")
                    # print(f"b.data: {b.data}")
                    return np.einsum("kd,knd->n", grad_val, b.data, optimize=True)

                # Test · Function  (rare but symmetrical) .............
                if b.role == "function" and a.role == "test":
                    grad_val = np.einsum("knd,kn->kd", b.data, b.coeffs, optimize=True)
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
            if integral.measure.domain_type == "ghost_edge":
                logger.info(f"Assembling ghost edge integral: {integral}")
                self._assemble_ghost_edge(integral, target)
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
        if self.backend == "python":
            logger.info(f"Assembling volume integral with python backend: {integral}")
            self._assemble_volume_python(integral, matvec)
        elif self.backend == "jit":
            logger.info(f"Assembling volume integral with jit backend: {integral}")
            self._assemble_volume_jit(integral, matvec)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}. Use 'python' or 'jit'.")
    
    # ----------------------------------------------------------------------
    def _assemble_volume_jit(self, integral: Integral, matvec):
        dbg = os.getenv("PYCUTFEM_JIT_DEBUG", "").lower() in {"1", "true", "yes"}

        # ------------------------------------------------------------------
        # 1. Get – or create – the JIT kernel runner
        # ------------------------------------------------------------------
        runner, ir = self._compile_backend(
            integral.integrand,          # the expression (no Form wrapper)
            self.dh, self.me
        )

        # ------------------------------------------------------------------
        # 2. Build (or reuse) the STATIC argument dictionary
        # ------------------------------------------------------------------
        q_order = self._find_q_order(integral)

        cache_key = (q_order, tuple(runner.param_order))   # hashable & unique enough
        if not hasattr(self, "_jit_static_cache"):
            self._jit_static_cache = {}
        if cache_key not in self._jit_static_cache:
            # (A) geometric factors      -------------------------------
            geo_args = self.dh.precompute_geometric_factors(q_order)

            # (B) element → global DOF map  &  node coordinates         ------
            mesh = self.me.mesh
            gdofs_map = np.vstack([
                self.dh.get_elemental_dofs(e) for e in range(mesh.n_elements)
            ]).astype(np.int32)
            node_coords = self.dh.get_all_dof_coords()

            # (C) basis / gradient / derivative tables (only what kernel needs)
            basis_args = _build_jit_kernel_args(
                ir, integral.integrand, self.me, q_order,
                dof_handler=self.dh,
                gdofs_map=gdofs_map,
                param_order=runner.param_order,
                pre_built= geo_args
            )

            static_args = {
                "gdofs_map":  gdofs_map,
                "node_coords": node_coords,
                **geo_args,
                **basis_args,
            }
            self._jit_static_cache[cache_key] = static_args
        else:
            static_args = self._jit_static_cache[cache_key]

        # ------------------------------------------------------------------
        # 3. Collect the up-to-date coefficient Functions
        # ------------------------------------------------------------------
        current_funcs = {
            f.name: f
            for f in _find_all(integral.integrand, (Function, VectorFunction))
        }

        # ------------------------------------------------------------------
        # 4. Execute the kernel via the runner
        # ------------------------------------------------------------------
        K_loc, F_loc, J_loc = runner(current_funcs, static_args)
        if dbg:
            print(f"[Assembler] kernel returned  K_loc {K_loc.shape}  "
                  f"F_loc {F_loc.shape}")

        # ------------------------------------------------------------------
        # 5.  Pure scalar functional  → never touches the global system
        # ------------------------------------------------------------------
        trial, test = _trial_test(integral.integrand)
        hook        = self.ctx["hooks"].get(type(integral.integrand))
        if trial is None and test is None:           # functional (hooked or not)
            if hook:                                 # accumulate if requested
                name = hook["name"]
                self.ctx.setdefault("scalar_results", {})[name] = 0.0
                self.ctx["scalar_results"][name] += J_loc.sum()
            return  

        # ------------------------------------------------------------------
        # 5. Scatter element contributions to the global system
        # ------------------------------------------------------------------
        mesh         = self.me.mesh
        gdofs_map    = static_args["gdofs_map"]
        n_dofs_local = self.me.n_dofs_local

        if self.ctx["rhs"]:                               # assembling a vector
            for e in range(mesh.n_elements):
                np.add.at(matvec, gdofs_map[e], F_loc[e])
        else:                                             # assembling a matrix
            data = np.empty(mesh.n_elements * n_dofs_local * n_dofs_local)
            rows = np.empty_like(data, dtype=np.int32)
            cols = np.empty_like(data, dtype=np.int32)

            for e in range(mesh.n_elements):
                gdofs = gdofs_map[e]
                r, c = np.meshgrid(gdofs, gdofs, indexing='ij')
                start = e * n_dofs_local * n_dofs_local
                end   = start + n_dofs_local * n_dofs_local

                rows[start:end] = r.ravel()
                cols[start:end] = c.ravel()
                data[start:end] = K_loc[e].ravel()

            K_coo = sp.coo_matrix(
                (data, (rows, cols)),
                shape=(self.dh.total_dofs, self.dh.total_dofs)
            )
            matvec += K_coo.tocsr()



    def _assemble_volume_python(self, integral: Integral, matvec):
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
                    # print(f"field: {f}, gref.shape: {g_ref.shape}, JiT.shape: {JiT.shape}")
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

    
    
    def _assemble_interface_python(self, intg: Integral, matvec): # New
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
                self.ctx.pop('global_dofs', None)
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
                        self.ctx['pos_eid'] = elem.id # New
                        self.ctx['neg_eid'] = elem.id # New
                        
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
                        self.ctx['pos_eid'] = elem.id
                        self.ctx['neg_eid'] = elem.id 
                        
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
                            self.ctx['global_dofs'] = self.dh.get_elemental_dofs(elem.id)
                            self.ctx['scalar_results'][hook['name']] += acc
                            log.debug(f"    Accumulated functional '{hook['name']}' for element {elem.id}") # New

        finally: # New
            # --- Final Context Cleanup --- # New
            for key in ('phi_val', 'normal', 'eid', 'x_phys', 'pos_eid', 'neg_eid', 'global_dofs'): # New
                self.ctx.pop(key, None) # New
            log.debug("Interface assembly finished. Context cleaned.") # New
    
    def _assemble_ghost_edge_python(self, intg: "Integral", matvec):
        """
        Assembles integrals over ghost facets using a side-aware cache.

        This routine correctly computes and caches basis function derivatives from
        both sides ('+' and '-') of each ghost edge. The UFL visitors for Jump,
        Avg, etc., are responsible for interpreting this side-aware cache to
        construct the final integrand, which couples the degrees of freedom
        from the adjacent elements. It also supports assembler hooks for assembling
        scalar-valued functionals.
        """

        rhs = self.ctx.get('rhs', False)
        mesh = self.me.mesh

        # 1. Determine derivative orders needed.
        derivs: set | None = intg.measure.metadata.get('derivs')
        if derivs is None:
            derivs = required_multi_indices(intg.integrand)
        if (0, 0) not in derivs:
            derivs = set(derivs) | {(0, 0)}

        # 2. Get edge set and level set.
        defined = intg.measure.defined_on
        if defined is None:                       # ← fallback: all ghost edges
            edge_ids = get_domain_bitset(mesh, 'edge', 'ghost')
        else:
            edge_ids = defined.to_indices()
        level_set = getattr(intg.measure, 'level_set', None)
        if level_set is None:
            raise ValueError("dGhost measure requires a 'level_set' callable.")

        qdeg = self._find_q_order(intg)
        fields = _all_fields(intg.integrand)
        
        # New: Check for assembler hooks to handle scalar functionals.
        trial, test = _trial_test(intg.integrand) # New
        hook = self.ctx['hooks'].get(type(intg.integrand)) if intg.integrand else self.ctx['hooks'].get(Function) # New
        is_functional = (hook is not None and trial is None and test is None) # New
        if is_functional: # New
            self.ctx.setdefault('scalar_results', {})[hook['name']] = 0.0 # New


        # 3. Main Loop over all candidate edges.
        for eid_edge in edge_ids:
            edge = mesh.edge(eid_edge)
            if edge.right is None:
                continue
            
            # edge_left = mesh.edges_list[edge.left]
            # edge_left_nodes = edge_left.nodes

            # edge_right = mesh.edges_list[edge.right]
            # edge_right_nodes = edge_right.nodes
            # pL00,pL01,pL10,pL11 = mesh.nodes_x_y_pos[list(edge_left_nodes)]
            # pR00,pR01,pR10,pR11 = mesh.nodes_x_y_pos[list(edge_right_nodes)]

            phi_left  = level_set(np.asarray(mesh.elements_list[edge.left].centroid()))
            phi_right = level_set(np.asarray(mesh.elements_list[edge.right].centroid()))
           
            if (phi_left < 0) == (phi_right < 0):
                continue

     
            pos_eid, neg_eid = (edge.left, edge.right) if phi_left >= 0 else (edge.right, edge.left)

            # 4. Setup for Quadrature and Local Assembly
            p0, p1 = mesh.nodes_x_y_pos[list(edge.nodes)]
            qpts_phys, qwts = line_quadrature(p0, p1, qdeg)
            normal_vec = edge.normal
            if np.dot(normal_vec, mesh.elements_list[pos_eid].centroid() - qpts_phys[0]) < 0:
                normal_vec *= -1.0

            loc_accumulator = 0.0 if is_functional else None # New
            
            # New: Only set up DOF mapping if we are not assembling a functional.
            if not is_functional: # New
                pos_dofs = self.dh.get_elemental_dofs(pos_eid)
                neg_dofs = self.dh.get_elemental_dofs(neg_eid)
                global_dofs = np.unique(np.concatenate((pos_dofs, neg_dofs)))
                pos_map = np.searchsorted(global_dofs, pos_dofs)   # e.g. [0,1,2,3,4,  8,9,10,11]
                neg_map = np.searchsorted(global_dofs, neg_dofs)   # e.g. [5,6,7,     12,13,14]

                if rhs:
                    loc_accumulator = np.zeros(len(global_dofs))
                else:
                    loc_accumulator = np.zeros((len(global_dofs), len(global_dofs)))

            # 5. Main Quadrature Loop
            
            for qp_phys, w in zip(qpts_phys, qwts):
                bv: dict[str, dict] = {'+': {}, '-': {}}
                for side, eid in (('+', pos_eid), ('-', neg_eid)):
                    xi, eta = transform.inverse_mapping(mesh, eid, qp_phys)
                    J = transform.jacobian(mesh, eid, (xi, eta))
                    J_inv = np.linalg.inv(J)
                    for fld in fields:
                        bv[side].setdefault(fld, {})
                        # ref = self.me._ref[fld]
                        # for alpha in derivs:
                        #     ref_val = ref.derivative(xi, eta, *alpha)
                        #     push = transform.map_deriv(alpha, J, J_inv)
                        #     phys_val = push(ref_val)
                        #     bv[side][fld][alpha] = phys_val
                        for alpha in derivs:
                            ox, oy = alpha
                            #------ref deriv zero padded
                            ref_val = self.me.deriv_ref(fld, xi, eta, ox, oy)
                            # -------- push-forward (axis-aligned map) ------------
                            # for curved elements we need to change this
                            sx = J_inv[0, 0] 
                            sy = J_inv[1, 1]
                            phys_val = ref_val * (sx ** ox) * (sy ** oy)   # (n_dofs,)
                            if is_functional:
                                # No union padding needed
                                bv[side][fld][alpha] = phys_val
                            else:
                                full_val = np.zeros(len(global_dofs))
                                if side == '+':
                                    full_val[pos_map] = phys_val
                                else:
                                    full_val[neg_map] = phys_val
                                # -------- cache for visitors -------------------------
                                bv[side][fld][alpha] = full_val

                eid_qp  = pos_eid if side == '+' else neg_eid
                ctx_data = {'basis_values': bv, 'normal': normal_vec,
                            'phi_val': level_set(qp_phys),
                            'eid': eid_qp,
                            'x_phys': qp_phys}
                ctx_data.update({'pos_eid': pos_eid, 'neg_eid': neg_eid})
                if not is_functional:
                    ctx_data.update({'global_dofs': global_dofs,
                                     'pos_map': pos_map,
                                     'neg_map': neg_map})
                self.ctx.update(ctx_data)
                integrand_val = self._visit(intg.integrand)
                loc_accumulator += w * integrand_val

            # 6. Scatter contributions.
            if is_functional: # New
                # If it's a functional, add the accumulated scalar to the results dictionary.
                self.ctx['scalar_results'][hook['name']] += loc_accumulator # New
                continue
            else: # New
                # Otherwise, scatter the local matrix/vector to the global system.
                if not rhs:
                    r, c = np.meshgrid(global_dofs, global_dofs, indexing="ij")
                    matvec[r, c] += loc_accumulator
                else:
                    np.add.at(matvec, global_dofs, loc_accumulator)

        # 7. Final Context Cleanup
        for k in ('basis_values', 'normal', 'phi_val', 'eid', 'x_phys', 'global_dofs', 'pos_map', 'neg_map'):
            self.ctx.pop(k, None)
    
    def _assemble_interface_jit(self, intg, matvec):
        """
        Assemble ∫_Γ ⋯ using the JIT backend.
        All kernel tables are sized (n_elems_total, …) so the kernel may
        iterate over *all* elements; non-cut rows are zero and contribute
        nothing.
        """

        dh, me = self.dh, self.me
        mesh   = me.mesh

        # 1.  BitSet of cut elements ------------------------------------------------
        cut_eids = (intg.measure.defined_on
                    if intg.measure.defined_on is not None
                    else mesh.element_bitset("cut"))

        if cut_eids.cardinality() == 0:            # nothing to do
            return

        # 2.  Geometric pre-compute --------------------------------------------------
        qdeg      = self._find_q_order(intg)
        level_set = intg.measure.level_set
        if level_set is None:
            raise ValueError("dInterface measure requires a level_set.")

        geo = dh.precompute_interface_factors(cut_eids, qdeg, level_set)

        # 3.  Full element-to-DOF map  (shape = n_elems_total × n_loc) --------------
        gdofs_map = np.vstack(
            [dh.get_elemental_dofs(eid) for eid in range(mesh.n_elements)]
        ).astype(np.int32)

        # 4.  gather coefficient Functions once ----------------------------
        current_funcs = {
            f.name: f
            for f in _find_all(intg.integrand, (Function, VectorFunction))
        }
        # also map parent VectorFunction names when we see a scalar component
        for f in list(current_funcs.values()):
            pv = getattr(f, "_parent_vector", None)
            if pv is not None:
                current_funcs.setdefault(pv.name, pv)

        # 5.  Compile kernel & build argument dict ----------------------------------
        runner, ir = self._compile_backend(intg.integrand, dh, me)

        basis_args = _build_jit_kernel_args(
            ir, intg.integrand, me, qdeg,
            dof_handler = dh,
            gdofs_map   = gdofs_map,
            param_order = runner.param_order,
            pre_built   = geo           # already sized n_elems_total
        )

        static_args = {k: v for k, v in geo.items() if k != 'eids'}
        static_args.update(basis_args)

        # 5.  Execute the kernel -----------------------------------------------------
        K, F, J = runner(current_funcs, static_args)

        # 6. scatter ONLY the cut rows --------------------------------------
        cut_eids = geo["eids"]                 # 1-D array of global ids
        K_cut = K[cut_eids]
        F_cut = F[cut_eids]
        J_cut = J[cut_eids] if J is not None else None

        gdofs_cut = gdofs_map[cut_eids]        # rows aligned with K_cut

        _scatter_element_contribs(
            K_cut, F_cut, J_cut,
            cut_eids, gdofs_cut,
            matvec, self.ctx, intg.integrand
        )

    def _assemble_interface(self, intg: Integral, matvec):
        """Assemble integrals over non-conforming interfaces defined by a level set."""
        if self.backend == "python":
            self._assemble_interface_python(intg, matvec)
        elif self.backend == "jit":
            self._assemble_interface_jit(intg, matvec)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}. Use 'python' or 'jit'.")
        
    def _assemble_ghost_edge_jit(self, intg: "Integral", matvec):
        """Assembles ghost-edge integrals using the JIT backend."""
        mesh = self.me.mesh
        dh, me = self.dh, self.me

        # 1. Get required derivatives, edge set, and level set
        derivs = required_multi_indices(intg.integrand)
        edge_ids = get_domain_bitset(mesh, 'edge', 'ghost')
        level_set = getattr(intg.measure, 'level_set', None)
        if level_set is None:
            raise ValueError("dGhost measure requires a 'level_set' callable.")
        qdeg = self._find_q_order(intg)

        if edge_ids.cardinality() == 0:
            return

        # 2. Precompute all side-aware geometric and basis factors
        geo_factors = self.dh.precompute_ghost_factors(edge_ids, qdeg, level_set, derivs)
        
        valid_eids = geo_factors.get('eids')
        if valid_eids is None or len(valid_eids) == 0:
            return

        # 3. Compile kernel
        runner, ir = self._compile_backend(intg.integrand, dh, me)

        

        # 4. Build kernel arguments
        # The precomputed factors are now the "basis tables"
        # We must also provide the DOF maps for coefficient padding
        pre_built_args = geo_factors.copy()
        
        # The ragged lists of dofs/maps need to be converted to something the
        # JIT helpers can use. A dense array is simplest.
        pre_built_args["gdofs_map"] = geo_factors["gdofs_map"]
        pre_built_args["pos_map"] =     geo_factors["pos_map"]
        pre_built_args["neg_map"] =     geo_factors["neg_map"]
        gdofs_map = pre_built_args["gdofs_map"]
        

        args = _build_jit_kernel_args(
            ir, intg.integrand, me, qdeg,
            dof_handler=dh,
            gdofs_map=gdofs_map, # This is now per-edge union of DOFs
            param_order=runner.param_order,
            pre_built=pre_built_args
        )

        # 5. Get current functions and execute kernel
        current_funcs = {
            f.name: f
            for f in _find_all(intg.integrand, (Function, VectorFunction))
        }
        # also map parent VectorFunction names when we see a scalar component
        for f in list(current_funcs.values()):
            pv = getattr(f, "_parent_vector", None)
            if pv is not None:
                current_funcs.setdefault(pv.name, pv)
        
        # The runner now gets per-edge arguments
        K_edge, F_edge, J_edge = runner(current_funcs, args)
        
        # 6. Scatter contributions
        _scatter_element_contribs(
            K_edge, F_edge, J_edge,
            valid_eids,       # The edge IDs
            gdofs_map,        # The per-edge union of DOFs
            matvec, self.ctx, intg.integrand
        )
    
    def _assemble_ghost_edge(self, intg: "Integral", matvec):
        """Assemble ghost edge integrals."""
        if self.backend == "python":
            self._assemble_ghost_edge_python(intg, matvec)
        elif self.backend == "jit":
            self._assemble_ghost_edge_jit(intg, matvec)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}. Use 'python' or 'jit'.")
