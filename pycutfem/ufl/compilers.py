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

# Symbolic building blocks
from pycutfem.ufl.expressions import (
    Constant,    TestFunction,   TrialFunction,
    VectorTestFunction, VectorTrialFunction,
    Function,    VectorFunction,
    Grad, DivOperation, Inner, Dot,
    Sum, Sub, Prod, Pos, Neg,Div
)
from pycutfem.ufl.forms import Equation
from pycutfem.ufl.measures import Integral

logger = logging.getLogger(__name__)

# ========================================================================
#  Tensor Containers for Symbolic Basis Functions
# ========================================================================
@dataclass(slots=True, frozen=True)
class VecOpInfo:
    """Container for vector basis functions φ (phi). Shape: (k, n)."""
    data: np.ndarray  # (num_components, n_loc_dofs)
    role: str = "none"  # "test" or "trial"

    def inner(self, other: "VecOpInfo") -> np.ndarray:
        """Computes inner product (u, v), returning an (n, n) matrix."""
        if self.data.shape[0] != other.data.shape[0]:
            raise ValueError("VecOpInfo component mismatch for inner product.")
        # sum over components (k), outer product over basis functions (n,m)
        return np.einsum("kn,km->nm", self.data, other.data, optimize=True)

    def dot_const(self, const: np.ndarray) -> np.ndarray:
        """Computes dot(v, c), returning an (n,) vector."""
        logger.debug(f"VecOpInfo.dot_const: const={const}, data.shape={self.data.shape}")
        const = np.asarray(const)
        if const.ndim != 1 or const.size != self.data.shape[0]:
            raise ValueError(f"Constant vector of size {const.size} is wrong length for VecOpInfo with {self.data.shape[0]} components.")
        return np.einsum("kn,k->n", self.data, const, optimize=True)
    # ========================================================================
    # Shape, len, and ndim methods
    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the shape of the data array."""
        return self.data.shape
    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the data array."""
        return self.data.ndim
    def __len__(self) -> int:
        """Returns the size of the first dimension (number of components)."""
        return self.data.shape[0] if self.data.ndim > 0 else 1
    def __mul__(self, other: Union[float, np.ndarray]) -> "VecOpInfo":
        """Element-wise multiplication with a scalar or vector."""
        if isinstance(other, (float, int)):
            return VecOpInfo(self.data * other, role=self.role)
        elif isinstance(other, np.ndarray):
            if other.ndim == 1 and other.size == self.data.shape[0]:
                return VecOpInfo(self.data * other[:, np.newaxis], role=self.role)
            else:
                raise ValueError(f"Cannot multiply VecOpInfo with array of shape {other.shape}.")
        else:
            raise TypeError(f"Unsupported multiplication type: {type(other)}")
    def __rmul__(self, other: Union[float, np.ndarray]) -> "VecOpInfo":
        return self.__mul__(other)
    def __add__(self, other: "VecOpInfo") -> "VecOpInfo":
        """Element-wise addition with another VecOpInfo."""
        if not isinstance(other, VecOpInfo):
            raise TypeError(f"Cannot add VecOpInfo to {type(other)}.")
        if self.data.shape != other.data.shape:
            raise ValueError("VecOpInfo shapes mismatch in addition.")
        return VecOpInfo(self.data + other.data, role=self.role)
    def __sub__(self, other: "VecOpInfo") -> "VecOpInfo":
        """Element-wise subtraction with another VecOpInfo."""
        if not isinstance(other, VecOpInfo):
            raise TypeError(f"Cannot subtract {type(other)} from VecOpInfo.")
        if self.data.shape != other.data.shape:
            raise ValueError("VecOpInfo shapes mismatch in subtraction.")
        return VecOpInfo(self.data - other.data, role=self.role)
    def __neg__(self) -> "VecOpInfo":
        """Element-wise negation of the VecOpInfo."""
        return VecOpInfo(-self.data, role=self.role)
    def info(self):
        """Return the type of the data array."""
        return f"VecOpInfo({self.data.dtype}, shape={self.data.shape}, role='{self.role}')"


@dataclass(slots=True, frozen=True)
class GradOpInfo:
    """Container for gradient of basis functions ∇φ. Shape: (k, n, d)."""
    data: np.ndarray  # (num_components, n_loc_dofs, spatial_dim)
    role: str = "none"

    def inner(self, other: GradOpInfo) -> np.ndarray:
        """Computes inner(∇u, ∇v) = ∫(∇u)T(∇v), returning an (n, n) matrix."""
        if not isinstance(other, GradOpInfo) or self.data.shape != other.data.shape:
            raise ValueError("Operands must be GradOpInfo of the same shape for inner product.")
        # sum over components (k) and spatial dims (d), outer product over basis funcs (n,m)
        return np.einsum("knd,kmd->nm", self.data, other.data, optimize=True)

    def dot_vec(self, vec: np.ndarray) -> VecOpInfo:
        """
        Computes the dot product with a constant vector over the SPATIAL dimension.
        This operation reduces the dimension and returns a VecOpInfo object.
        dot(∇v, c) -> ∇v ⋅ c
        """
        vec = np.asarray(vec)
        if vec.ndim != 1 or vec.size != self.shape[-1]: # Must match spatial dim `d`
            raise ValueError(f"Input vector of size {vec.size} cannot be dotted with spatial dimension of size {self.shape[-1]}.")
        
        # einsum("knd,d->kn", ...)
        result_data = np.einsum("knd,d->kn", self.data, vec, optimize=True)
        return VecOpInfo(result_data, role=self.role)

    # ========================================================================
    # Dunder methods for SCALING Operations
    # ========================================================================

    def __mul__(self, other: Union[float, int, np.ndarray]) -> GradOpInfo:
        """
        Performs SCALING and always returns a new GradOpInfo object.
        This operator preserves the shape (k, n, d).
        """
        if isinstance(other, (float, int)):
            # Case 1: Scaling by a single scalar
            new_data = self.data * other
            
        elif isinstance(other, np.ndarray):
            other = np.asarray(other)
            if other.ndim == 0:
                # Case 1b: Scaling by a 0-D array (e.g., np.array(2))
                new_data = self.data * other
            elif other.ndim == 1 and other.size == self.shape[0]:
                # Case 2: Scaling by component (vector of size k)
                # Reshape (k,) -> (k, 1, 1) to broadcast over (k, n, d)
                new_data = self.data * other[:, np.newaxis, np.newaxis]
            elif other.ndim == 1 and other.size == self.shape[1]:
                # Case 3: Scaling by location (vector of size n, e.g. a density field rho)
                # Reshape (n,) -> (1, n, 1) to broadcast over (k, n, d)
                new_data = self.data * other[np.newaxis, :, np.newaxis]
            else:
                 raise ValueError(f"Cannot scale GradOpInfo(shape={self.shape}) with array of shape {other.shape}.")
        else:
            return NotImplemented

        return GradOpInfo(new_data, role=self.role)

    def __rmul__(self, other: Union[float, int, np.ndarray]) -> GradOpInfo:
        return self.__mul__(other)
        
    def __neg__(self) -> GradOpInfo:
        return GradOpInfo(-self.data, role=self.role)

    def __add__(self, other: GradOpInfo) -> GradOpInfo:
        if not isinstance(other, GradOpInfo) or self.data.shape != other.data.shape:
            raise ValueError("Operands must be GradOpInfo of the same shape for addition.")
        return GradOpInfo(self.data + other.data, role=self.role)

    def __sub__(self, other: GradOpInfo) -> GradOpInfo:
        return self.__add__(-other)

    # --- Helper properties ---
    @property
    def shape(self) -> Tuple[int, ...]: return self.data.shape
    @property
    def ndim(self) -> int: return self.data.ndim
    def __repr__(self): return f"GradOpInfo(shape={self.data.shape}, role='{self.role}')"

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
        self.ctx: Dict[str, Any] = {}
        self._basis_cache: Dict[str, Dict[str, np.ndarray]] = {}
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
            Div: self._visit_Div
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
    def _visit_Neg(self, n): return -self._visit(n.operand)
    def _visit_Pos(self, n): return self._visit(n.operand)
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
        # Use repr for detailed logging of symbolic expression parts
        logger.debug(f"Entering _visit_Prod for  ('{n.a!r}' * '{n.b!r}') on {'RHS' if self.ctx['rhs'] else 'LHS'}, a.info={getattr(a, 'info', None)}, b.info={getattr(b, 'info', None)}")
        logger.debug(f"  a: {type(a)} (role={role_a}), b: {type(b)} (role={role_b})")


        # ------------------------------------------------------------------
        # scalar × scalar  (Constant * Constant  or numeric literals)
        # ------------------------------------------------------------------
        if np.isscalar(a) and np.isscalar(b):
            return a * b
        

        # ------------------------------------------------------------------
        # scalar x vector multiplication
        # ------------------------------------------------------------------
        if np.isscalar(a) and isinstance(b, (VecOpInfo, GradOpInfo, np.ndarray)):
            return a * b
        if np.isscalar(b) and isinstance(a, (VecOpInfo, GradOpInfo, np.ndarray)):
            return b * a

        # --- RHS: one operand MUST be a value, the other a Test function basis ---
        if self.ctx["rhs"]:
            # (known) VecOpInfo  *  (test) VecOpInfo  → length-n vector
            if isinstance(a, VecOpInfo) and a.role == "function" and \
            isinstance(b, VecOpInfo) and b.role == "test":

                # scalar field (k = 1)  .................  p_k · div(v)
                if a.data.shape[0] == 1:
                    f_val = np.sum(a.data)                 # p(x_q)
                    return f_val * b.data[0]               # p · div(v)

                # vector field (k = 2)  ..................  (u_k − u_n , v)
                return np.einsum("kn,kn->n", a.data, b.data, optimize=True)

            # symmetric orientation
            if isinstance(b, VecOpInfo) and b.role == "function" and \
            isinstance(a, VecOpInfo) and a.role == "test":

                if b.data.shape[0] == 1:
                    f_val = np.sum(b.data)
                    return f_val * a.data[0]
                return np.einsum("kn,kn->n", b.data, a.data, optimize=True)

            # scalar Constant or numpy value times test basis 
            if np.isscalar(a) and isinstance(b, VecOpInfo) and b.role == "test":
                return a * b.data[0]
            if np.isscalar(b) and isinstance(a, VecOpInfo) and a.role == "test":
                return b * a.data[0]
            

        # --- LHS: matrix assembly ---
        else: 
            # # Case 1: One operand is a scalar value (from Constant or evaluated expression like inner/dot)
            # if np.isscalar(a) or (isinstance(a, np.ndarray) and a.ndim == 0):
            #     return a * b
            # if np.isscalar(b) or (isinstance(b, np.ndarray) and b.ndim == 0):
            #     return b * a

            # Case 2: Both operands represent fields. Unwrap their data for outer product.
            # ✓  (test , trial)   →  rows = test, cols = trial
            if isinstance(a, VecOpInfo) and isinstance(b, VecOpInfo):
                # v (test)  ·  w (trial / function)
                if a.role == "test"  and b.role in {"trial", "function"}:
                    # M_nm = Σ_k  a_k,n  *  b_k,m
                    return np.einsum("kn,km->nm", a.data, b.data, optimize=True)
                if b.role == "test"  and a.role in {"trial", "function"}:
                    return np.einsum("kn,km->nm", b.data, a.data, optimize=True)
            
            # fallback (same-role or plain arrays)
            # if a_vec.ndim == 1 and b_vec.ndim == 1:
            #     return np.outer(a_vec, b_vec)

        raise TypeError(f"Unsupported product 'type(a)={type(a)}, type(b)={type(b)}'{n.a} * {n.b}' for {'RHS' if self.ctx['rhs'] else 'LHS'}")


        

    def _visit_Dot(self, n: Dot):
        a = self._visit(n.a)
        b = self._visit(n.b)
        a_data = a.data if isinstance(a, (VecOpInfo, GradOpInfo)) else a
        b_data = b.data if isinstance(b, (VecOpInfo, GradOpInfo)) else b
        logger.debug(f"Entering _visit_Dot for types {type(a)} . {type(b)}")

        def rhs():
            # ------------------------------------------------------------------
            # RHS  •  dot(  Function ,  Test  )   or   dot( Test , Function )
            # ------------------------------------------------------------------
            if  isinstance(a, VecOpInfo) and isinstance(b, VecOpInfo):
                # if a.role == "function" and b.role == "test":
                #     # f_k,n · v_k,n  →  Σ_k f_k,n v_k,n   (length-n vector for RHS)
                #     return np.einsum("km,kn->n", a.data, b.data, optimize=True)
                # if b.role == "function" and a.role == "test":
                #     return np.einsum("km,kn->n", b.data, a.data, optimize=True)
                 #  func · test
                if a.role == "function" and b.role == "test":
                    u_val = np.sum(a.data, axis=1)                 # <- 1-D (k,)
                    return np.einsum("k,kn->n", u_val, b.data, optimize=True)

                #  test · func
                if b.role == "function" and a.role == "test":
                    u_val = np.sum(b.data, axis=1)                 # <- 1-D (k,)
                    return np.einsum("k,kn->n", u_val, a.data, optimize=True)

            # ------------------------------------------------------------------
            # Function · Function   (needed for  dot(grad(u_k), u_k)  on RHS)
            # ------------------------------------------------------------------
            if  isinstance(a, GradOpInfo) and isinstance(b, VecOpInfo) \
            and a.role == b.role == "function":
                u_val   = np.sum(b.data, axis=1)                # (d,)
                data_fk = np.einsum("knd,d->kn", a.data, u_val) # keep basis-index n
                return VecOpInfo(data_fk, role="function")
            # Constant (numpy 1-D) · test VecOpInfo  → length-n vector
            if isinstance(a, np.ndarray) and a.ndim == 1 and \
            isinstance(b, VecOpInfo) and b.role == "test":
                return b.dot_const(a)

            if isinstance(b, np.ndarray) and b.ndim == 1 and \
            isinstance(a, VecOpInfo) and a.role == "test":
                return a.dot_const(b)

        if self.ctx["rhs"]:
            result = rhs()
            if result is not None:
                return result
            else:
                raise TypeError(f"Unsupported dot product for RHS: '{n.a} . {n.b}'")
        
        # Dot product between a basis field and a numerical vector
        if isinstance(a, (VecOpInfo, GradOpInfo)) and isinstance(b, np.ndarray): return a.dot_const(b) if a.ndim==2 else a.dot_vec(b)
        if isinstance(b, (VecOpInfo, GradOpInfo)) and isinstance(a, np.ndarray): return b.dot_const(a) if b.ndim==2 else b.dot_vec(a)
        

        if isinstance(a, VecOpInfo) and isinstance(b, VecOpInfo):
            logger.debug(f"visit dot: Both operands are VecOpInfo: {a.role} . {b.role}")
            if a.role == "test" and b.role in {"trial", "function"}:
                return np.dot(a_data.T, b_data)  # test . trial
            elif b.role == "test" and a.role in {"trial", "function"}:
                return np.dot(b_data.T, a_data)  # tiral . test
        
        # case grad(u) . u_k
        if isinstance(a, GradOpInfo) and ((isinstance(b, VecOpInfo) \
            and (b.role == "function" )) 
            ):

            # velocity value at this quadrature point
            u_val = np.sum(b.data, axis=1)                 # (d,)

            # works for k = 1 (scalar) and k = 2 (vector) alike
            data  = np.einsum("knd,d->kn", a.data, u_val, optimize=True)

            return VecOpInfo(data, role=a.role)
        
        # ------------------------------------------------------------------
        # Case:  VectorFunction · Grad(⋅)      u_k · ∇w
        # ------------------------------------------------------------------
        if isinstance(a, VecOpInfo) and isinstance(b, GradOpInfo) \
        and a.role == "function":

            # 1. velocity value at the current quadrature point
            u_val = np.sum(a.data, axis=1)                  # (d,)  —   u_d(ξ)

            # 2. dot with each gradient row   w_{k,n} = Σ_d u_d ∂_d φ_{k,n}
            #    Works for both scalar (k = 1) and vector (k = 2) targets.
            data  = np.einsum("d,knd->kn", u_val, b.data, optimize=True)

            # 3. The role is inherited from the Grad operand (trial / test / function)
            return VecOpInfo(data, role=b.role)
        
        # ------------------------------------------------------------------
        # Case:  Grad(Function) · Vec(Trial)      (∇u_k) · u
        # ------------------------------------------------------------------
        if isinstance(a, GradOpInfo) and a.role == "function" \
        and isinstance(b, VecOpInfo)  and b.role == "trial":

            # (1)  value of ∇u_k at this quad-point
            grad_val = np.sum(a.data, axis=1)          # (k,d)

            # (2)  w_i,n = Σ_d (∇u_k)_i,d  *  φ_{u,d,n}
            data = np.einsum("kd,kn->kn", grad_val, b.data, optimize=True)

            return VecOpInfo(data, role="trial")
        
        # ------------------------------------------------------------------
        # Case:  Vec(Function) · Grad(Trial)      u_k · ∇u
        # ------------------------------------------------------------------
        if isinstance(a, VecOpInfo)  and a.role == "function" \
        and isinstance(b, GradOpInfo) and b.role == "trial":

            u_val = np.sum(a.data, axis=1)             # (d,)
            data  = np.einsum("d,knd->kn", u_val, b.data, optimize=True)
            return VecOpInfo(data, role="trial")
        
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
            return a.inner(b)

        # One is a basis, the other a numerical tensor value (RHS or complex LHS)
        if isinstance(a, GradOpInfo) and isinstance(b, np.ndarray): return a.contracted_with_tensor(b)
        if isinstance(b, GradOpInfo) and isinstance(a, np.ndarray): return b.contracted_with_tensor(a)
        
        # RHS: both are functions, result is scalar integral
        if self.ctx['rhs'] and isinstance(a, VecOpInfo) and isinstance(b, VecOpInfo):
             return np.sum(a.data * b.data)

        raise TypeError(f"Unsupported inner product '{n.a} : {n.b}'")
        
    # Visitor dispatch
    def _visit(self, node): return self._dispatch[type(node)](node)

    # ======================= ASSEMBLY CORE ============================
    def _assemble_form(self, form, target):
        if form is None: return
        integrals = form.integrals if hasattr(form, "integrals") else [form]
        for ing in integrals:
            if ing.measure.domain_type != "volume":
                logger.error(f"Unsupported integral type: {ing.measure.domain_type}")
                raise NotImplementedError
            self._assemble_volume(ing, target)

    def _assemble_volume(self, integral: Integral, matvec):
        mesh = self.me.mesh
        # Use a higher quadrature order for safety, esp. with mixed orders
        q_order = self.qorder or self.me.mesh.poly_order + max(self.me._field_orders.values()) + 1
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
        
        self.ctx.pop("eid", None)

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
        K = K.tolil()
        K[rows, :] = 0
        K[:, rows] = 0
        K[rows, rows] = 1.0
        
        # Set values in the RHS vector
        F[rows] = vals