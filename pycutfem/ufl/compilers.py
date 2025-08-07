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

from matplotlib.pylab import f

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
    ElementWiseConstant, Derivative, Transpose,
    CellDiameter, NormalComponent,
    Restriction, Power
)
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.measures import Integral
from pycutfem.ufl.quadrature import PolynomialDegreeEstimator
from pycutfem.ufl.helpers import VecOpInfo, GradOpInfo, required_multi_indices,_all_fields,_find_all,_trial_test
from pycutfem.fem.transform import map_deriv
from pycutfem.ufl.analytic import Analytic
from pycutfem.ufl.helpers_jit import  _build_jit_kernel_args, _scatter_element_contribs, _stack_ragged
from pycutfem.ufl.helpers_geom import (
    phi_eval, clip_triangle_to_side, fan_triangulate, map_ref_tri_to_phys, corner_tris
)



logger = logging.getLogger(__name__)
_INTERFACE_TOL = 1.0e-12 # New




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
            Derivative        : self._visit_Derivative,
            Transpose: self._visit_Transpose,
            CellDiameter: self._visit_CellDiameter,
            NormalComponent: self._visit_NormalComponent,
            Restriction: self._visit_Restriction,
            Power: self._visit_Power
        }

    # ============================ PUBLIC API ===============================
    def assemble(self, eq: Equation, bcs: Union[Mapping, Iterable, None] = None):
        ndofs = self.dh.total_dofs
        K = sp.lil_matrix((ndofs, ndofs))
        F = np.zeros(ndofs)

        # Assemble LHS if it is provided.
        if eq.a is not None:
            logger.info("Assembling LHS...")
            self.ctx["rhs"] = False
            self._assemble_form(eq.a, K)

        # Assemble RHS if it is provided.
        if eq.L is not None:
            logger.info("Assembling RHS vector F...")
            self.ctx["rhs"] = True
            self._assemble_form(eq.L, F)
        
        # --- THE FIX ---
        # Apply boundary conditions AFTER both LHS and RHS have been assembled.
        # This block is now outside of the `if eq.L` check.
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
            side = self.ctx.get("side")
            if side not in ('+', '-'):
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
    
    def _visit_Restriction(self, n: Restriction):
        """
        Evaluates the operand only if the current element's tag matches the
        restriction tag. Otherwise, returns a zero-like value.
        """
        # 1. Get the current element's tag.
        #    The element ID is reliably in the context during volume/interface assembly.
        eid = self.ctx.get("eid")
        if eid is None:
            # This can happen in contexts where 'eid' is not defined.
            # We proceed, but the check might fail gracefully later.
            elem_tag = None
        else:
            in_domain = n.domain[eid] if hasattr(n.domain, "__getitem__") else eid in n.domain

        # 2. Check if the element is in the active domain.
        if in_domain:
            # If it matches, proceed as if the Restriction node wasn't there.
            return self._visit(n.operand)
        else:
            # If it does not match, return a "zero" value that has the same
            # structure (shape, role) as the real result would have.
            # This is crucial for subsequent operations like dot, inner, etc.
            
            # We can achieve this by visiting the operand and then nullifying it.
            result = self._visit(n.operand)
            
            if isinstance(result, (VecOpInfo, GradOpInfo)):
                # Use the overloaded multiplication to create a zeroed-out copy.
                return result * 0.0
            elif isinstance(result, np.ndarray):
                # For raw numpy arrays, just return a zero-filled array of the same shape.
                return np.zeros_like(result)
            else: # for scalars
                return 0.0
    
    def _visit_NormalComponent(self, n:NormalComponent):
        return self._visit_FacetNormal(FacetNormal())[n.idx]
    
    def _visit_Power(self, n: Power):
        """
        Handles the power operation, which is not a standard UFL operation.
        It can be used to raise a field to a constant power.
        """
        # Visit the base and exponent separately.
        base = self._visit(n.a)
        exponent = self._visit(n.b)

        # Handle custom VecOpInfo and GradOpInfo types explicitly
        if isinstance(base, (VecOpInfo, GradOpInfo)):
            new_data = base.data ** exponent
            # Return a new instance of the same class with the new data
            # This uses getattr for robustness, as VecOpInfo lacks 'coeffs'
            return type(base)(data=new_data, role=base.role, coeffs=getattr(base, 'coeffs', None))

        # Ensure the exponent is a scalar (constant).
        if not np.isscalar(exponent):
            raise ValueError("Exponent must be a scalar constant.")

        # Raise the base to the power of the exponent.
        return base ** exponent


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
            old_side  = self.ctx.get("side", None)

            # + side
            self.ctx['phi_val'] = 1.0
            self.ctx["side"]    = '+' 
            if 'pos_eid' in self.ctx:
                self.ctx['eid'] = self.ctx['pos_eid']
            v_pos = self._visit(dpos)

            # – side
            self.ctx['phi_val'] = -1.0
            self.ctx["side"]    = '-'
            if 'neg_eid' in self.ctx:
                self.ctx['eid'] = self.ctx['neg_eid']
            v_neg = self._visit(dneg)

            # restore
            self.ctx['phi_val'] = phi_old
            if eid_old is None:
                self.ctx.pop('eid', None)
            else:
                self.ctx['eid'] = eid_old
            if old_side is None:
                self.ctx.pop('side', None)
            else:
                self.ctx['side'] = old_side

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
    

    def _visit_CellDiameter(self, node):
        eid = self.ctx["eid"]          # set by every volume / interface loop
        return self.me.mesh.element_char_length(eid)

    
    # ================= VISITORS: ALGEBRAIC ==========================
    def _visit_Sum(self, n): return self._visit(n.a) + self._visit(n.b)
    def _visit_Sub(self, n):
        a = self._visit(n.a)
        b = self._visit(n.b)
        return a - b
    
    def _visit_Transpose(self, node:Transpose):
        mat = self._visit(node.A)               # recurse
        # NumPy ndarray  → just use .T
        if isinstance(mat, np.ndarray):
            return mat.T
        if isinstance(mat, GradOpInfo):
            return mat.transpose()            
        # VecOpInfo (our own small tensor wrapper)
        if isinstance(mat, VecOpInfo):
            return mat.transpose()              # implement below
        raise TypeError(f"Cannot transpose object of type {type(mat)}")

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
            elif isinstance(b, np.ndarray) and isinstance(a, VecOpInfo):
                return a * b
            elif isinstance(a, np.ndarray) and isinstance(b, VecOpInfo):
                return b * a
        
        raise TypeError(f"Unsupported product 'type(a)={type(a)}, type(b)={type(b)}'{n.a} * {n.b}' for {'RHS' if self.ctx['rhs'] else 'LHS'}"
                        f" for roles a={getattr(a, 'role', None)}, b={getattr(b, 'role', None)}")

        

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
        if isinstance(a, (VecOpInfo)) and isinstance(b, np.ndarray): 
            return a.dot_const_vec(b)
        elif isinstance(b, (VecOpInfo)) and isinstance(a, np.ndarray): 
            return b.dot_const_vec(a)
        elif isinstance(a, (GradOpInfo)) and isinstance(b, np.ndarray):
            result = a.dot_vec(b)
            # print(f"visit dot: GradOpInfo . result: {result}, result shape: {result.shape}, role: {getattr(result, 'role', None)}")
            return result
        elif isinstance(b, (GradOpInfo)) and isinstance(a, np.ndarray):
            return b.dot_vec(a)
        

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
    def _assemble_form(self, form, target):
        """Accept a Form object and iterate through its integrals."""
        from pycutfem.ufl.forms import Form
        # from pycutfem.ufl.measures import Integral

        # This guard clause correctly handles one-sided equations (e.g., L=None or a=None)
        if form is None or not isinstance(form, Form):
            return

        for integral in form.integrals:
            if not isinstance(integral, Integral):
                continue
            
            # --- The rest of the dispatching logic remains the same ---
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
            if integral.measure.domain_type == "exterior_facet":
                logger.info(f"Assembling exterior-facet integral: {integral}")
                self._assemble_boundary_edge(integral, target)
                continue

            logger.warning(f"Skipping unsupported integral type: {integral.measure.domain_type}")


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
    
    def _hook_for(self, expr):
        hooks = self.ctx.get("hooks", {})
        # print(f"expr={expr}, hooks={hooks}")
        if expr is None:
            return hooks.get(Function)           # default
        if expr in hooks:                        # ← object hash & equality!
            return hooks[expr]
        return hooks.get(type(expr))             # fallback by type
    
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
        # if a level-set was attached to dx → do cut-cell assembly in Python path
        if getattr(integral.measure, "level_set", None) is not None:
            # CUT-VOLUME integral
            if self.backend == "jit":
                self._assemble_volume_cut_jit(integral, matvec)
            else:
                # existing pure-Python path (already working from earlier work)
                self._assemble_volume_cut_python(integral, matvec)
            return
        # … otherwise the existing back-end selection …
        if self.backend == "python":
            self._assemble_volume_python(integral, matvec)
        elif self.backend == "jit":
            self._assemble_volume_jit(integral, matvec)
        else:
            raise ValueError("Unsupported backend.")
    
    # ----------------------------------------------------------------------
    def _assemble_volume_jit(self, integral: Integral, matvec):
        """
        Assembles a volume integral using the JIT backend, correctly handling
        element subsets and ensuring argument lists match the specific integral.
        """
        dbg = os.getenv("PYCUTFEM_JIT_DEBUG", "").lower() in {"1", "true", "yes"}
        mesh = self.me.mesh

        # 1. Determine the exact subset of elements for this integral.
        defined_on = integral.measure.defined_on
        if defined_on is not None:
            element_ids = np.asarray(defined_on.to_indices(), dtype=np.int32) if hasattr(defined_on, "to_indices") else np.flatnonzero(np.asarray(defined_on)).astype(np.int32)
        else:
            element_ids = np.arange(mesh.n_elements, dtype=np.int32)

        if element_ids.size == 0:
            return # Nothing to assemble.

        # 2. Collect the up-to-date coefficient functions from the expression FIRST.
        # This aligns with the calling sequence of other robust JIT assemblers.
        current_funcs = self._get_data_functions_objs(integral)

        # 3. Compile the backend for the CURRENT integral to get the specific runner and IR.
        # This ensures we have the correct param_order for this exact expression.
        runner, ir = self._compile_backend(
            integral.integrand,
            self.dh, self.me
        )

        # 4. Build the static arguments required by this specific kernel.
        # We build these fresh every time to avoid caching collisions.
        q_order = self._find_q_order(integral)

        # (A) Get full-mesh geometric factors and then SLICE them for the subset.
        geo_args_all = self.dh.precompute_geometric_factors(q_order)
        pre_built = {
            "qp_phys": geo_args_all["qp_phys"][element_ids],
            "qw":      geo_args_all["qw"][element_ids],
            "detJ":    geo_args_all["detJ"][element_ids],
            "J_inv":   geo_args_all["J_inv"][element_ids],
            "normals": geo_args_all["normals"][element_ids],
            "h_arr":   geo_args_all["h_arr"][element_ids],
            "phis":    None if geo_args_all["phis"] is None else geo_args_all["phis"][element_ids],
        }

        # (B) Build the DOF map for the subset.
        gdofs_map = np.vstack([
            self.dh.get_elemental_dofs(e) for e in element_ids
        ]).astype(np.int32)

        # (C) Build all other required tables (basis, etc.) based on the correct param_order.
        basis_args = _build_jit_kernel_args(
            ir, integral.integrand, self.me, q_order,
            dof_handler=self.dh,
            gdofs_map=gdofs_map,
            param_order=runner.param_order, # Use the param_order from the CURRENT runner
            pre_built=pre_built
        )

        # (D) Finalize the dictionary of static arguments for the kernel runner.
        static_args = {
            "gdofs_map":  gdofs_map,
            "node_coords": self.dh.get_all_dof_coords(),
            **pre_built,
            **basis_args,
        }

        # 5. Execute the kernel via the runner.
        K_loc, F_loc, J_loc = runner(current_funcs, static_args)
        if dbg:
            print(f"[Assembler] kernel returned K_loc {K_loc.shape}, F_loc {F_loc.shape}")

        # 6. Handle scalar functional result or scatter the element contributions.
        hook = self._hook_for(integral.integrand)
        if self._functional_calculate(integral, J_loc, hook):
            return

        _scatter_element_contribs(
            K_loc, F_loc, J_loc,
            element_ids=element_ids,
            gdofs_map=gdofs_map,
            matvec=matvec,
            ctx=self.ctx,
            integrand=integral.integrand,
            hook=hook
        )
        



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
        
        hook = self._hook_for(intg.integrand)
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
        Assemble integrals over ghost facets (pure Python path).

        This version:
        * trusts the provided ghost edge ids except for minimal validity checks;
        * populates a side-aware basis cache for both (+) and (–);
        * guarantees all derivative multi-indices needed by the form are present;
        * orients the facet normal from (–) to (+);
        * supports scalar-valued functionals via assembler hooks.
        """
        # --------------------------------------------------------- basics
        rhs  = self.ctx.get("rhs", False)
        mesh = self.me.mesh

        # ---- derivative orders we need (robust) -------------------------
        md = intg.measure.metadata or {}
        derivs = set(md.get("derivs", set())) | set(required_multi_indices(intg.integrand))
        derivs |= {(0, 0)}  # always include values

        # Close the set up to the maximum total order present.
        # This prevents KeyError for cross terms like (1,1) in Hessian energies.
        max_total = max((ox + oy for (ox, oy) in derivs), default=0)
        for p in range(max_total + 1):
            for ox in range(p + 1):
                derivs.add((ox, p - ox))

        # ---- which edges -------------------------------------------------
        defined = intg.measure.defined_on
        if defined is None:
            edge_set = mesh.edge_bitset("ghost")
            edge_ids = (edge_set.to_indices() if hasattr(edge_set, "to_indices")
                        else list(edge_set))
        else:
            edge_ids = defined.to_indices()

        level_set = getattr(intg.measure, "level_set", None)
        if level_set is None:
            raise ValueError("dGhost measure requires a 'level_set' callable.")

        qdeg   = self._find_q_order(intg)
        fields = _all_fields(intg.integrand)

        # ---- hook / functional support ----------------------------------
        trial, test = _trial_test(intg.integrand)
        hook = self._hook_for(intg.integrand)
        is_functional = (hook is not None and trial is None and test is None)
        if is_functional:
            self.ctx.setdefault("scalar_results", {})[hook["name"]] = 0.0

        # -------------------------------------------------------- main loop
        for eid_edge in edge_ids:
            e = mesh.edge(eid_edge)
            # ghost facets must be interior (two neighbors)
            if e.right is None:
                continue

            # Keep edges in the narrow band (at least one CUT neighbor),
            # or edges already tagged as ghost_* by the mesh.
            lt = mesh.elements_list[e.left].tag
            rt = mesh.elements_list[e.right].tag
            etag = str(getattr(e, "tag", ""))
            if not (("cut" in (lt, rt)) or etag.startswith("ghost")):
                continue

            # (+) and (–) sides: decide by the left-element φ sign
            phiL = level_set(np.asarray(mesh.elements_list[e.left].centroid()))
            pos_eid, neg_eid = (e.left, e.right) if phiL >= 0 else (e.right, e.left)

            # Quadrature and oriented normal
            p0, p1 = mesh.nodes_x_y_pos[list(e.nodes)]
            qpts_phys, qwts = line_quadrature(p0, p1, qdeg)

            normal_vec = e.normal
            if np.dot(normal_vec, mesh.elements_list[pos_eid].centroid() - qpts_phys[0]) < 0:
                normal_vec = -normal_vec  # from (–) to (+)

            # Local accumulator & maps
            if is_functional:
                loc_accumulator = 0.0
                global_dofs = pos_map = neg_map = None
            else:
                pos_dofs = self.dh.get_elemental_dofs(pos_eid)
                neg_dofs = self.dh.get_elemental_dofs(neg_eid)
                global_dofs = np.unique(np.concatenate((pos_dofs, neg_dofs)))
                pos_map = np.searchsorted(global_dofs, pos_dofs)
                neg_map = np.searchsorted(global_dofs, neg_dofs)
                if rhs:
                    loc_accumulator = np.zeros(len(global_dofs))
                else:
                    loc_accumulator = np.zeros((len(global_dofs), len(global_dofs)))

            # ------------------------------------------------- quadrature loop
            for xq, w in zip(qpts_phys, qwts):
                # Build the side-aware cache for this qp
                bv = {"+": {}, "-": {}}

                for side, eid in (("+", pos_eid), ("-", neg_eid)):
                    xi, eta = transform.inverse_mapping(mesh, eid, xq)
                    J      = transform.jacobian(mesh, eid, (xi, eta))
                    J_inv  = np.linalg.inv(J)

                    # axis-aligned / structured assumption
                    sx, sy = J_inv[0, 0], J_inv[1, 1]

                    for fld in fields:
                        fld_cache = {}
                        for ox, oy in derivs:
                            ref_val  = self.me.deriv_ref(fld, xi, eta, ox, oy)  # (n_loc,)
                            phys_val = ref_val * (sx ** ox) * (sy ** oy)

                            if is_functional:
                                # (n_loc,) is fine for functionals
                                fld_cache[(ox, oy)] = phys_val
                            else:
                                # pad to union with zeros on the other side
                                full = np.zeros(len(global_dofs))
                                if side == "+":
                                    full[pos_map] = phys_val
                                else:
                                    full[neg_map] = phys_val
                                fld_cache[(ox, oy)] = full

                        bv[side][fld] = fld_cache

                # Context for visitors
                self.ctx.update({
                    "basis_values": bv,
                    "normal": normal_vec,
                    "phi_val": level_set(xq),
                    "x_phys": xq,
                    "pos_eid": pos_eid,
                    "neg_eid": neg_eid,
                })
                if not is_functional:
                    self.ctx.update({
                        "global_dofs": global_dofs,
                        "pos_map": pos_map,
                        "neg_map": neg_map,
                    })

                # Evaluate integrand and accumulate
                integrand_val = self._visit(intg.integrand)
                loc_accumulator += w * integrand_val

            # ----------------------------------------------- scatter / finalize
            if is_functional:
                self.ctx["scalar_results"][hook["name"]] += loc_accumulator
            else:
                if not rhs:
                    r, c = np.meshgrid(global_dofs, global_dofs, indexing="ij")
                    matvec[r, c] += loc_accumulator
                else:
                    np.add.at(matvec, global_dofs, loc_accumulator)

        # ----------------------------------------------- cleanup context keys
        for k in ("basis_values", "normal", "phi_val", "x_phys",
                "global_dofs", "pos_map", "neg_map", "pos_eid", "neg_eid"):
            self.ctx.pop(k, None)

    

    def _assemble_interface_jit(self, intg, matvec):
        """
        Assemble interface terms (∫_Γ …) with the JIT back‑end **using
        cut‑only tables**.

        All geometry/basis tables coming from
        `DofHandler.precompute_interface_factors` are sized
        ``n_cut_elems × …`` and contain *only* the rows that correspond to
        the global element ids listed in ``geo["eids"]``.  We therefore:

        1. build a DOF‑map of the same length,
        2. feed those aligned arrays to the kernel, and
        3. scatter the kernel output with the matching global‑id list.
        """
        dh, me  = self.dh, self.me
        mesh    = me.mesh

        # ------------------------------------------------------------------
        # 1. Which elements carry an interface?  (BitSet → array of ids)
        # ------------------------------------------------------------------
        cut_bs = (intg.measure.defined_on
                if intg.measure.defined_on is not None
                else mesh.element_bitset("cut"))

        if cut_bs.cardinality() == 0:          # nothing to do
            return

        # ------------------------------------------------------------------
        # 2. Pre‑compute geometry for the interface elements only
        # ------------------------------------------------------------------
        qdeg      = self._find_q_order(intg)
        level_set = intg.measure.level_set
        if level_set is None:
            raise ValueError("dInterface measure requires a level_set.")

        geo = dh.precompute_interface_factors(cut_bs, qdeg, level_set)
        cut_eids = geo["eids"].astype(np.int32)      # 1‑D array, len = n_cut

        # ------------------------------------------------------------------
        # 3. Element‑to‑DOF map  (shape = n_cut × n_loc)
        # ------------------------------------------------------------------
        gdofs_map = np.vstack(
            [dh.get_elemental_dofs(eid) for eid in cut_eids]
        ).astype(np.int32)
        geo["gdofs_map"] = gdofs_map

        # ------------------------------------------------------------------
        # 4. Gather coefficient Functions once
        # ------------------------------------------------------------------
        current_funcs = self._get_data_functions_objs(intg)

        # ------------------------------------------------------------------
        # 5. Compile kernel & build static argument dict
        # ------------------------------------------------------------------

        runner, ir = self._compile_backend(intg.integrand, dh, me)

        basis_args = _build_jit_kernel_args(
            ir, intg.integrand, me, qdeg,
            dof_handler = dh,
            gdofs_map   = gdofs_map,
            param_order = runner.param_order,
            pre_built   = geo
        )

        static_args = {k: v for k, v in geo.items() if k != "eids"}
        static_args.update(basis_args)

        # ------------------------------------------------------------------
        # 6. Execute kernel  → element buffers K/F/J
        # ------------------------------------------------------------------
        K_cut, F_cut, J_cut = runner(current_funcs, static_args)

        # ------------------------------------------------------------------
        # 7. Scatter the contributions from the cut elements
        # ------------------------------------------------------------------
        hook = self._hook_for(intg.integrand)
        if self._functional_calculate(intg, J_cut, hook): return

        _scatter_element_contribs(
            K_cut, F_cut, J_cut,
            cut_eids, gdofs_map,
            matvec, self.ctx, intg.integrand,
            hook = hook,          # scalar‑functional hook, if any
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
        derivs = required_multi_indices(intg.integrand) | {(0, 0)}
        edge_ids = (intg.measure.defined_on
                if intg.measure.defined_on is not None
                else mesh.edge_bitset('ghost'))
        level_set = getattr(intg.measure, 'level_set', None)
        if level_set is None:
            raise ValueError("dGhost measure requires a 'level_set' callable.")
        qdeg = self._find_q_order(intg)

        if edge_ids.cardinality() == 0:
            return

        # 2. Precompute all side-aware geometric and basis factors
        geo_factors = self.dh.precompute_ghost_factors(edge_ids, qdeg, level_set, derivs)
        
        valid_eids = geo_factors.get('eids')
        print(f"len(valid_eids)={len(valid_eids)}")
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
        current_funcs = self._get_data_functions_objs(intg)
        
        # The runner now gets per-edge arguments
        K_edge, F_edge, J_edge = runner(current_funcs, args)
        hook = self._hook_for(intg.integrand)
        if self._functional_calculate(intg, J_edge, hook): return

        # 6. Scatter contributions
        _scatter_element_contribs(
            K_edge, F_edge, J_edge,
            valid_eids,       # The edge IDs
            gdofs_map,        # The per-edge union of DOFs
            matvec, self.ctx, intg.integrand,
            hook = hook,  # Pass the hook for scalar functionals
        )
    
    def _assemble_ghost_edge(self, intg: "Integral", matvec):
        """Assemble ghost edge integrals."""
        if self.backend == "python":
            self._assemble_ghost_edge_python(intg, matvec)
        elif self.backend == "jit":
            self._assemble_ghost_edge_jit(intg, matvec)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}. Use 'python' or 'jit'.")


    # inside FormCompiler -------------------------------------------------
    def _assemble_boundary_edge_python(self, intg: Integral, matvec):
        """
        Assemble ∫_Γ  f  dS   over a *set* of boundary edges  Γ  (Neumann BC,
        surface traction in elasticity, etc.).  Works for bilinear, linear
        and scalar functional forms – scalar forms can be captured with the
        same “assembler_hooks” mechanism used for interfaces / ghost edges.
        """
        if self.backend == "jit":
            raise NotImplementedError("dS + JIT not wired yet")

        rhs   = self.ctx.get("rhs", False)
        mesh  = self.me.mesh
        qdeg  = self._find_q_order(intg)
        hook  = self._hook_for(intg.integrand)
        trial, test = _trial_test(intg.integrand)
        is_functional = hook and trial is None and test is None
        if is_functional:
            self.ctx.setdefault("scalar_results", {})[hook["name"]] = 0.0

        # decide which edges we visit -------------------------------------
        defined = intg.measure.defined_on
        edge_ids = (defined.to_indices() if defined is not None
                    else np.fromiter((e.right is None for e in mesh.edges_list), bool).nonzero()[0])

        fields = _all_fields(intg.integrand)

        if edge_ids.size == 0: raise ValueError(
            f"Integral {intg} has no defined edges. "
            "Check the measure or the mesh.")
        for gid in edge_ids:
            edge   = mesh.edge(gid)
            eid    = edge.left                      # the unique owner element
            p0, p1 = mesh.nodes_x_y_pos[list(edge.nodes)]
            qp, qw = line_quadrature(p0, p1, qdeg)  # *weights already in phys. space*

            # outward normal of the *element*:
            n = edge.normal.copy()
            # if np.dot(n, np.asarray(mesh.elements_list[eid].centroid()) - qp[0]) < 0:
            #     n *= -1.0
            self.ctx["normal"] = n                  # constant along that edge

            # precompute element dofs once
            gdofs = self.dh.get_elemental_dofs(eid)

            # ------------------------------------------------------------------
            loc = None
            for x_phys, w in zip(qp, qw):
                xi, eta = transform.inverse_mapping(mesh, eid, x_phys)
                JiT     = np.linalg.inv(transform.jacobian(mesh, eid, (xi, eta))).T

                # basis cache ---------------------------------------------------
                self._basis_cache.clear()
                for f in fields:
                    self._basis_cache[f] = {
                        "val" : self.me.basis      (f, xi, eta),
                        "grad": self.me.grad_basis (f, xi, eta) @ JiT
                    }

                # context for visitors
                self.ctx.update({"eid": eid, "x_phys": x_phys})

                val = self._visit(intg.integrand)    # scalar, vec, or matrix
                loc = (w * np.asarray(val)           # **NO detJ re-multiplication!**
                    if loc is None else loc + w * np.asarray(val))

            # ---------------- scatter -----------------------------------------
            if rhs and test:                   # linear form  (vector assemble)
                np.add.at(matvec, gdofs, loc)
            elif not rhs and trial and test:   # bilinear form (matrix assemble)
                r, c = np.meshgrid(gdofs, gdofs, indexing="ij")
                matvec[r, c] += loc
            elif is_functional:               # hooked scalar functional
                # print('-'*50)
                # print(f"loc.shape: {loc.shape}, type: {type(loc)}")
                if isinstance(loc, VecOpInfo):    # collapse (k,n) → scalar
                    loc = np.sum(loc.data, axis=1)
                self.ctx["global_dofs"] = gdofs
                self.ctx["scalar_results"][hook["name"]] += loc

        # tidy
        for k in ("eid", "x_phys", "normal", "global_dofs"):
            self.ctx.pop(k, None)
    def _get_data_functions_objs(self, intg: Integral):
        """
        Extracts all Function and VectorFunction objects from the integrand.
        This is used to gather the current state of coefficient functions
        before executing the JIT kernel.
        """
        coeffs = (Function, VectorFunction)
        current = {f.name: f
                   for f in _find_all(intg.integrand, coeffs)
                   if not getattr(f, "is_test", False)
                   and not getattr(f, "is_trial", False)}
        
        # Add parent vectors if they look like real coefficient vectors
        for f in list(current.values()):
            pv = getattr(f, "_parent_vector", None)
            if pv is not None and hasattr(pv, "name"):
                current.setdefault(pv.name, pv)

        return current
    
    def _functional_calculate(self, intg: Integral, J_loc, hook):
        trial, test = _trial_test(intg.integrand)
        is_functional = (trial is None and test is None)
        flag = False
        if is_functional:
            # J_loc is (n_edges,) or scalar; collapse to plain float
            if J_loc is not None:
                if J_loc.ndim > 1:
                    total = np.sum(J_loc, axis=0)  # (n_edges,) → scalar
                if J_loc.ndim == 1:
                    total = J_loc.sum()
            name  = hook["name"]               # retrieved earlier via _hook_for
            scal  = self.ctx.setdefault("scalar_results", {})
            scal[name] =  total
            flag = True
        return flag                            # 🔑  skip the scatter stage

    def _assemble_boundary_edge_jit(self, intg: Integral, matvec):
        mesh, dh, me = self.me.mesh, self.dh, self.me

        # 1. pick edges -----------------------------------------------
        edge_set = (intg.measure.defined_on
                    if intg.measure.defined_on is not None
                    else mesh.get_domain_bitset(intg.measure.tag, entity="edge"))
        if edge_set.cardinality() == 0:
            return

        # 2. geometry tables ------------------------------------------
        derivs = required_multi_indices(intg.integrand) | {(0, 0)}  # always include values
        if (0, 0) not in derivs:
            derivs |= {(0, 0)}
        qdeg = self._find_q_order(intg)
        print(f"[Assembler: boundary edge JIT] Using quadrature degree: {qdeg}")
        geo  = dh.precompute_boundary_factors(edge_set, qdeg, derivs)

        valid = geo["eids"]
        if valid.size == 0:
            return

        # 3. kernel compilation ---------------------------------------
        runner, ir = self._compile_backend(intg.integrand, dh, me)

        args = _build_jit_kernel_args(
            ir, intg.integrand, me, qdeg,
            dof_handler = dh,
            gdofs_map   = geo["gdofs_map"],
            param_order = runner.param_order,
            pre_built   = geo
        )

        # 4. up-to-date coefficient Functions -------------------------
        #    • ignore test/trial symbols
        #    • add parent vectors only if they look like real coefficient vectors
        current = self._get_data_functions_objs(intg)

        # 5. execute and scatter --------------------------------------
        K_loc, F_loc, J_loc = runner(current, args)


        hook = self._hook_for(intg.integrand)
        if self._functional_calculate(intg, J_loc, hook): return
            
        _scatter_element_contribs(
            K_loc, F_loc, J_loc,
            element_ids=geo["eids"],                 # rows ≡ edge ids
            gdofs_map=geo["gdofs_map"],
            matvec=matvec, 
            ctx=self.ctx, 
            integrand=intg.integrand,
            hook=hook,
        )
    def _assemble_boundary_edge(self, intg: Integral, matvec):
        """Assemble integrals over boundary edges."""
        if self.backend == "python":
            self._assemble_boundary_edge_python(intg, matvec)
        elif self.backend == "jit":
            self._assemble_boundary_edge_jit(intg, matvec)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}. Use 'python' or 'jit'.")
    # --- CUT-CELL helpers -------------------------------------------------

    def _physical_tri_quadrature(self, tri_coords, q_order):
        """
        Map the reference-triangle quadrature to the physical triangle with vertices tri_coords.
        Returns (qpoints_phys, qweights_phys). Weights are *physical* areas.
        """
        v0, v1, v2 = map(np.asarray, tri_coords)
        qp_ref, qw_ref = volume('tri', q_order)     # reference rule (r,s) on Δ(0,0)-(1,0)-(0,1)
        J = np.column_stack((v1 - v0, v2 - v0))     # 2x2 affine map
        detJ = abs(np.linalg.det(J))
        qp_phys = (qp_ref @ J.T) + v0               # (nq,2)
        qw_phys = qw_ref * detJ                     # area scaling → physical weights
        return qp_phys, qw_phys
    
    def _integrate_on_cut_element(self, eid: int, integral, level_set, q_order: int, side: str):
        """
        Integrate over the physical part of element `eid` by clipping each geometric
        corner-triangle against phi=0, then sub-triangulating and integrating each piece.
        Returns the local element vector/matrix (shape matches self.ctx['rhs']).
        """
        mesh   = self.me.mesh
        fields = _all_fields(integral.integrand)
        rhs    = self.ctx.get('rhs', False)

        # local accumulator
        loc = (np.zeros(self.me.n_dofs_local) if rhs
            else np.zeros((self.me.n_dofs_local, self.me.n_dofs_local)))

        # 1) split geometry into corner triangles (tri stays as-is; quad → two tris)
        elem = mesh.elements_list[eid]
        tri_local, corner_ids = corner_tris(mesh, elem)

        # 2) reference rule on the unit triangle
        qp_ref, qw_ref = volume("tri", q_order)

        # 3) process each geometric triangle
        for loc_tri in tri_local:
            v_ids   = [corner_ids[i] for i in loc_tri]
            v_coords = mesh.nodes_x_y_pos[v_ids]  # (3,2)
            v_phi    = np.array([phi_eval(level_set, xy) for xy in v_coords], dtype=float)

            # clip to requested side and fan-triangulate each polygon
            polygons = clip_triangle_to_side(v_coords, v_phi, side=side)
            for poly in polygons:
                for A, B, C in fan_triangulate(poly):
                    qp_phys, qw_phys = map_ref_tri_to_phys(A, B, C, qp_ref, qw_ref)

                    # 4) quadrature loop in *physical* space (weights are physical)
                    for x_phys, w in zip(qp_phys, qw_phys):
                        # reference point of the *parent element* that owns x_phys
                        xi, eta = transform.inverse_mapping(mesh, eid, x_phys)
                        J       = transform.jacobian(mesh, eid, (xi, eta))
                        JiT     = np.linalg.inv(J).T

                        # cache bases at this point (zero-padded across mixed fields)
                        self._basis_cache.clear()
                        for f in fields:
                            self._basis_cache[f] = {
                                "val" : self.me.basis(f, xi, eta),
                                "grad": self.me.grad_basis(f, xi, eta) @ JiT
                            }

                        # context for UFL visitors
                        self.ctx["eid"]     = eid
                        self.ctx["x_phys"]  = x_phys
                        self.ctx["phi_val"] = phi_eval(level_set, x_phys)

                        integrand_val = self._visit(integral.integrand)
                        # IMPORTANT: 'w' already includes the area Jacobian of the sub-triangle.
                        loc += w * integrand_val

        return loc

    def _assemble_volume_cut_python(self, integral, matvec):
        """
        Assemble a volume integral restricted by the 'side' of a level set:
            dx(level_set=phi, metadata={'side': '+/-'})
        - full elements on the selected side: standard reference rule
        - cut elements: clipped sub-triangle physical quadrature
        """
        mesh      = self.me.mesh
        level_set = integral.measure.level_set
        if level_set is None:
            raise ValueError("Cut-cell volume assembly requires measure.level_set")

        side    = integral.measure.metadata.get('side', '+')  # '+' → phi>0, '-' → phi<0
        q_order = self._find_q_order(integral)
        rhs     = self.ctx.get("rhs", False)
        fields  = _all_fields(integral.integrand)

        # --- 1) classify (fills tags & caches)
        inside_ids, outside_ids, cut_ids = mesh.classify_elements(level_set)

        # Reference rule & scatter util
        qp_ref, qw_ref = volume(mesh.element_type, q_order)

        def _scatter(eid, loc):
            gdofs = self.dh.get_elemental_dofs(eid)
            if rhs:
                np.add.at(matvec, gdofs, loc)
            else:
                r, c = np.meshgrid(gdofs, gdofs, indexing="ij")
                matvec[r, c] += loc

        # --- 2) full elements on the selected side → standard reference rule
        full_eids = outside_ids if side == '+' else inside_ids
        for eid in full_eids:
            loc = (np.zeros(self.me.n_dofs_local) if rhs
                else np.zeros((self.me.n_dofs_local, self.me.n_dofs_local)))

            for (xi, eta), w in zip(qp_ref, qw_ref):
                J    = transform.jacobian(mesh, eid, (xi, eta))
                detJ = abs(np.linalg.det(J))
                JiT  = np.linalg.inv(J).T

                self._basis_cache.clear()
                for f in fields:
                    self._basis_cache[f] = {
                        "val" : self.me.basis(f, xi, eta),
                        "grad": self.me.grad_basis(f, xi, eta) @ JiT
                    }
                self.ctx["eid"]    = eid
                self.ctx["x_phys"] = transform.x_mapping(mesh, eid, (xi, eta))
                integrand_val      = self._visit(integral.integrand)
                loc += w * detJ * integrand_val

            _scatter(eid, loc)

        # --- 3) cut elements → clipped sub-triangles
        for eid in cut_ids:
            loc = self._integrate_on_cut_element(eid, integral, level_set, q_order, side=side)
            _scatter(eid, loc)

        # cleanup
        for key in ("eid", "x_phys", "phi_val"):
            self.ctx.pop(key, None)

    
    def _assemble_volume_cut_jit(self, intg, matvec):
        """
        Assemble ∫_{Ω∩{φ▹0}} (...) dx with the JIT back-end.

        Strategy:
        1) classify elements w.r.t. φ
        2) assemble full elements on the selected side with the *standard* JIT volume
        3) assemble cut elements with element-specific (clipped) quadrature
        """
        mesh, dh, me = self.me.mesh, self.dh, self.me
        qdeg   = self._find_q_order(intg)
        level_set = intg.measure.level_set
        side   = intg.measure.metadata.get('side', '+')   # '+' → φ>0, '-' → φ<0

        # --- 1) classification (must be called; sets element tags & caches)
        inside_ids, outside_ids, cut_ids = mesh.classify_elements(level_set)

        # --- compile kernel once
        runner, ir = self._compile_backend(intg.integrand, dh, me)

        # helper to run kernel on a subset of elements with prebuilt geo
        def _run_subset(eids: np.ndarray, prebuilt: dict):
            if eids.size == 0:
                return
            gdofs_map = np.vstack([dh.get_elemental_dofs(e) for e in eids]).astype(np.int32)
            # kernel wants node_coords in the signature, provide it once
            prebuilt = dict(prebuilt)
            prebuilt["gdofs_map"]  = gdofs_map
            prebuilt["node_coords"] = dh.get_all_dof_coords()  # required in param_order

            static_args = _build_jit_kernel_args(
                ir, intg.integrand, me, qdeg,
                dof_handler=dh,
                gdofs_map=gdofs_map,
                param_order=runner.param_order,
                pre_built=prebuilt
            )
            current_funcs = self._get_data_functions_objs(intg)
            K_loc, F_loc, J_loc = runner(current_funcs, static_args)

            hook = self._hook_for(intg.integrand)
            _scatter_element_contribs(
                K_loc, F_loc, J_loc, eids, gdofs_map,
                matvec, self.ctx, intg.integrand, hook=hook
            )

        # --- 2) full elements on the selected side → standard volume tables sliced
        full_ids = np.asarray(outside_ids if side == '+' else inside_ids, dtype=np.int32)
        if full_ids.size:
            # include level_set so 'phis' is populated (still fine if None)
            geo_all = dh.precompute_geometric_factors(qdeg, level_set)

            # slice what the kernel signature expects
            prebuilt_full = {
                "qp_phys": geo_all["qp_phys"][full_ids],
                "qw":      geo_all["qw"][full_ids],
                "detJ":    geo_all["detJ"][full_ids],
                "J_inv":   geo_all["J_inv"][full_ids],
                "normals": geo_all["normals"][full_ids],
                # 'phis' may be None if level_set was None; keep it as None in that case
                "phis":    None if geo_all["phis"] is None else geo_all["phis"][full_ids],
            }
            _run_subset(full_ids, prebuilt_full)

        # --- 3) cut elements → clipped triangles (physical weights); detJ := 1
        if len(cut_ids):
            from pycutfem.ufl.helpers import required_multi_indices
            derivs = required_multi_indices(intg.integrand) | {(0, 0)}

            # precomputed, element-specific qp / qw / J_inv  (+ basis tables b_* / dxy_*)
            geo_cut = dh.precompute_cut_volume_factors(
                mesh.element_bitset("cut"), qdeg, derivs, level_set, side=side
            )
            cut_eids = np.asarray(geo_cut.get("eids", []), dtype=np.int32)
            if cut_eids.size:
                # ensure detJ is present and neutral (if your helper has not added it)
                if "detJ" not in geo_cut:
                    geo_cut["detJ"] = np.ones_like(geo_cut["qw"])
                _run_subset(cut_eids, geo_cut)

        # cleanup context (if you stored any debug keys)
        for k in ("eid", "x_phys", "phi_val"):
            self.ctx.pop(k, None)





