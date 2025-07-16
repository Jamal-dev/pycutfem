import re
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, Set
from pycutfem.ufl.expressions import Expression, Derivative

import logging
# Setup logging
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
        # case for Constant with dim = 1 and also normal vector
        logger.debug(f"VecOpInfo.dot_const: const={const}, data.shape={self.data.shape}")
        const = np.asarray(const)
        if const.ndim != 1 or const.size != self.data.shape[0]:
            raise ValueError(f"Constant vector of size {const.size} is wrong length for VecOpInfo with {self.data.shape[0]} components.")
        return np.einsum("kn,k->n", self.data, const, optimize=True)
    def dot_const_vec(self, const: np.ndarray) -> "VecOpInfo":
        """Computes dot(v, c), returning an (n,) vector."""
        # case for Constant with dim = 1 and also normal vector
        logger.debug(f"VecOpInfo.dot_const: const={const}, data.shape={self.data.shape}")
        const = np.asarray(const)
        if const.ndim != 1 or const.size != self.data.shape[0]:
            raise ValueError(f"Constant vector of size {const.size} is wrong length for VecOpInfo with {self.data.shape[0]} components.")
        #data =  np.einsum("kn,k->n", self.data, const, optimize=True)
        data = self.data.T @ const  # fast BLAS-2
        return VecOpInfo(data[np.newaxis,:], role=self.role)
    def dot_grad(self, grad: "GradOpInfo") -> "VecOpInfo":
        """
        Computes dot(v, ∇u) for a vector basis function v and gradient ∇u.
        Returns a new VecOpInfo with shape (k, n).
        """
        if not isinstance(grad, GradOpInfo):
            raise TypeError(f"Expected GradOpInfo, got {type(grad)}.")
        if self.data.shape[0] != grad.data.shape[-1]:
            raise ValueError(f"VecOpInfo {self.shape} and GradOpInfo {grad.shape} must have the same number of components.")
        
        # Case 1: Function · Grad(Trial) or Trial · Grad(Function)
        # 1. value of each velocity component at the current Q-point
        if self.role == "function" and grad.role == "trial":
            # Case:  Function · Grad(Trial)      u_k · ∇u_trial
            # (1)  value of u_k at this quad-point
            u_val = np.sum(self.data, axis=1)          # shape (k,)  —   u_k(ξ)
            # (2)  w_i,n = Σ_d u_d ∂_d φ_{k,n}
            data = np.einsum("d,kld->kl", u_val, grad.data, optimize=True)
            return VecOpInfo(data, role=grad.role)
        elif self.role == "trial" and grad.role == "function":
            # Case:  Trial · Grad(Function)      u_trial · ∇u_k
            # (1)  value of u_trial at this quad-point
            grad_val = np.einsum("knd,kn->kd", grad.data, grad.coeffs, optimize=True) # (k, d)  —   ∇u_k(ξ) (2,2)
            # (2)  w_i,n = Σ_d ∂_d φ_{k,n} u_d
            data = np.einsum("dl,kd->kl", self.data, grad_val, optimize=True)
            return VecOpInfo(data, role=self.role)
        raise NotImplementedError(f"VecOpInfo.dot_grad not implemented for role {self.role} and GradOpInfo role {grad.role}.")
    
    def dot_vec(self, vec: "VecOpInfo") -> "VecOpInfo":
        """
        Computes dot(u, v) for a vector basis function u and a other vector basis.
        Returns a new VecOpInfo with shape (n,).
        """
        # print(f" a.shape = {self.data.shape}, b.shape = {vec.data.shape}")
        if not isinstance(vec, VecOpInfo):
            raise TypeError(f"Expected VecOpInfo, got {type(vec)}.")
        if vec.shape[0] != self.data.shape[0]:
            raise ValueError(f"Input vector of shape {vec.shape} cannot be dotted with VecOpInfo of shape {self.data.shape}.")
        
        # case 1 function dot test
        if vec.role == "test" and self.role == "function": # rhs time derivative term
            func_values_at_qp = np.sum(self.data, axis=1)  # Shape (k,)
            return np.einsum("k,kn->n", func_values_at_qp, vec.data, optimize=True)
            
        # case 2 trial dot test and test dot trial
        if self.role == "trial" and vec.role == "test":
            # lhs mass matrix term
            # If self is trial and vec is test, we return a VecOpInfo with shape (m, n)
            return np.einsum("km,kn->mn", vec.data , self.data, optimize=True)
        elif self.role == "test" and vec.role == "trial":
            return np.einsum("km,kn->mn", self.data, vec.data, optimize=True)
        
        # case 3 trial and function
        if self.role == "trial" and vec.role == "function":
            u_values = np.sum(vec.data, axis=1)  # Shape (k,)
            data = np.einsum("kn,k->n", self.data, u_values, optimize=True)
            return VecOpInfo(data, role=self.role)
        elif self.role == "function" and vec.role == "trial":
            # If self is function and vec is trial, we return a VecOpInfo with shape (m, n)
            u_values = np.sum(self.data, axis=1)
            data = np.einsum("k,kn->n", u_values, vec.data, optimize=True)
            return VecOpInfo(data, role=vec.role)
        # case 4 function and function
        if self.role == "function" and vec.role == "function":
            u_values = np.sum(self.data, axis=1)  # Shape (k,)
            v_values = np.sum(vec.data, axis=1)  # Shape (k,)
            return np.dot(u_values, v_values)  # scalar result for rhs
        raise NotImplementedError(f"VecOpInfo.dot_vec not implemented for roles {self.role} and {vec.role}.")
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
        elif isinstance(other, VecOpInfo):
            if self.data.shape != other.data.shape:
                raise ValueError("VecOpInfo shapes mismatch in multiplication.")
            if self.role == "trial" and other.role == "test":
                # Case: Trial * Test , outer product case
                return np.einsum("km,kn->mn", other.data , self.data, optimize=True)
            elif self.role == "test" and other.role == "trial":
                # Case: Test * Trial , outer product case
                return np.einsum("km,kn->mn", self.data, other.data, optimize=True)
            elif self.role == "function" and other.role == "function":
                # Case: Function * Function , inner product case
                u_vals = np.sum(self.data, axis=1)  # Shape (k,)
                v_vals = np.sum(other.data, axis=1)  # Shape (k,)
                return np.dot(u_vals, v_vals)  # scalar result for rhs
            elif self.role == "trial" and other.role == "function":
                # Case: Trial * Function , dot product case
                u_vals = np.sum(other.data, axis=1)
                data = np.einsum("kn,k->n", self.data, u_vals, optimize=True)
                return VecOpInfo(data, role=self.role)
            elif self.role == "function" and other.role == "trial":
                # Case: Function * Trial , dot product case
                u_vals = np.sum(self.data, axis=1)
                data = np.einsum("k,kn->n", u_vals, other.data, optimize=True)
                return VecOpInfo(data, role=other.role)
            else:
                raise NotImplementedError(f"VecOpInfo multiplication not implemented for roles {self.role} and {other.role}.")
                
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
    def __repr__(self) -> str:
        """String representation of the VecOpInfo."""
        return f"VecOpInfo(shape={self.data.shape}, role='{self.role}')"


@dataclass(slots=True, frozen=True)
class GradOpInfo:
    """Container for gradient of basis functions ∇φ. Shape: (k, n, d)."""
    data: np.ndarray  # (num_components, n_loc_dofs, spatial_dim)
    role: str = "none"
    coeffs: np.ndarray = field(default=None)

    def inner(self, other: "GradOpInfo") -> np.ndarray:
        """Computes inner(∇u, ∇v) = ∫(∇u)T(∇v), returning an (n, n) matrix."""
        if not isinstance(other, GradOpInfo) or self.data.shape != other.data.shape:
            raise ValueError("Operands must be GradOpInfo of the same shape for inner product.")
        # sum over components (k) and spatial dims (d), outer product over basis funcs (n,m)
        return np.einsum("knd,kmd->nm", self.data, other.data, optimize=True)
    

        
        

    def dot_vec(self, vec: np.ndarray) -> VecOpInfo:
        """
        Computes the dot product with a constant vector or VecOpInfo over the SPATIAL dimension.
        This operation reduces the dimension and returns a VecOpInfo object.
        dot(∇v, c) -> ∇v ⋅ c
        """
        
        if isinstance(vec, (VecOpInfo)): # this part is until trial grad(u) dot u_k  ((∇u) · u_k)
            # role = self.role
            # if vec.role == "trial":
            #     role = vec.role
            # if vec.data.shape[0] != self.data.shape[-1]:
            #     raise ValueError(f"Cannot dot GradOpInfo {self.shape} with VecOpInfo of shape {vec.data.shape}.")
            # result_data = np.einsum("ijk,kl->ij", self.data, vec.data, optimize=True)
            # return VecOpInfo(result_data, role=role)
            if self.role == "function" and vec.role == "trial": # introducing a new branch
                # Case:  Grad(Function) · Vec(Trial)      (∇u_k) · u
                # (k, d)   =  Σ_i  u_{k,i}  ∂_d φ_i   – true ∇u_k at this quad-point
                grad_val = np.einsum("knd,kn->kd", self.data, self.coeffs, optimize=True)

                # (k, n)   =  Σ_d  (∇u_k)_d  φ_{u,d,n}     – (∇u_k)·δu  for every trial dof n
                # data = grad_val @ vec.data                  # BLAS-2, same as einsum("kd,dn->kn")
                data = grad_val @ vec.data
                
                return VecOpInfo(data, role=vec.role)
            
            if self.role == "trial" and vec.role == "function": # introducing a new branch
                # Case:  Grad(Trial) · Vec(Function)      (∇u_trial) · u_k
                u_val = np.sum(vec.data, axis=1)          # (k,d) ∫ u_k dx ) 
                data = np.einsum("kld,d->kl", self.data, u_val, optimize=True)
                return VecOpInfo(data, role=self.role)
            if self.role == "function" and vec.role == "function":
                # Case:  Grad(Function) · Vec(Function)      (∇u_k) · u_k
                # (1)  value of ∇u_k at this quad-point
                grad_val = np.einsum("knd,kn->kd", self.data, self.coeffs, optimize=True)
                u_val = np.sum(vec.data, axis=1)
                w_val = np.einsum("kd,d->k", grad_val, u_val, optimize=True)
                return w_val # rhs, so return a 1D array

            
        
        if isinstance(vec, np.ndarray) and vec.ndim == 1: # dot product with a constant vector
            if self.role == "function" and self.coeffs is not None:
                # Case:  Grad(Function) · Const      (∇u_k) · c
                # (1)  value of ∇u_k at this quad-point
                grad_val = np.einsum("knd,kn->kd", self.data, self.coeffs, optimize=True)
                # (2)  w_i,n = Σ_d (∇u_k)_d c_d φ_{k,n}
                result_data = np.einsum("kd,d->k", grad_val, vec, optimize=True)
                return result_data # returns a 1D array
            result_data = np.einsum("knd,d->kn", self.data, vec, optimize=True)
            return VecOpInfo(result_data, role=self.role)
        raise NotImplementedError(f"dot_vec of GradOpInfo not implemented for role {self.role}, GradOpInfo  and type {type(vec)} with role: {vec.role}.")

    # ========================================================================
    # Dunder methods for SCALING Operations
    # ========================================================================

    def __mul__(self, other: Union[float, int, np.ndarray]) -> "GradOpInfo":
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

    def __rmul__(self, other: Union[float, int, np.ndarray]) -> "GradOpInfo":
        return self.__mul__(other)
        
    def __neg__(self) -> "GradOpInfo":
        return GradOpInfo(-self.data, role=self.role)

    def __add__(self, other: "GradOpInfo") -> "GradOpInfo":
        if not isinstance(other, GradOpInfo) or self.data.shape != other.data.shape:
            raise ValueError("Operands must be GradOpInfo of the same shape for addition.")
        return GradOpInfo(self.data + other.data, role=self.role)

    def __sub__(self, other: "GradOpInfo") -> "GradOpInfo":
        return self.__add__(-other)

    # --- Helper properties ---
    @property
    def shape(self) -> Tuple[int, ...]: return self.data.shape
    @property
    def ndim(self) -> int: return self.data.ndim
    def __repr__(self): return f"GradOpInfo(shape={self.data.shape}, role='{self.role}')"
    def info(self):
        """Return the type of the data array."""
        return f"GradOpInfo({self.data.dtype}, shape={self.data.shape}, role='{self.role}')"

# ----------------------------------------------------------------------
# New definition (helpers.py)
# ----------------------------------------------------------------------
MultiIndex = Tuple[int, int]  # (α_x, α_y)

def required_multi_indices(expr: "Expression") -> Set[MultiIndex]:
    """
    Collect each distinct derivative order (α_x, α_y) that occurs anywhere
    in *expr*.

    Works with
        • Derivative(f, order=(αx, αy))      – new API
        • nested Derivative(f, dir) chains   – old API

    A node can appear more than once in the graph (e.g. a VectorFunction
    points to its two scalar components).  We keep a *seen* set so the
    walk terminates even on graphs with back-references.
    """
    out:  Set[MultiIndex] = set()   # final result
    seen: Set[int]        = set()   # id(node) → already visited

    def _walk(node: "Expression", acc_x: int = 0, acc_y: int = 0):
        # ---- NEW: break potential cycles --------------------------------
        nid = id(node)
        if nid in seen:
            return
        seen.add(nid)

        # ---- Derivative node --------------------------------------------
        if isinstance(node, Derivative):
            # -- new API ---------------------------------------------------
            if hasattr(node, "order"):
                ox, oy = node.order
                _walk(node.f, acc_x + ox, acc_y + oy)
                return
            # -- old API (single direction) --------------------------------
            dir_ = getattr(node, "component_index", None)
            if dir_ == 0:
                _walk(node.f, acc_x + 1, acc_y)
            elif dir_ == 1:
                _walk(node.f, acc_x, acc_y + 1)
            else:                       # unrecognised → just recurse
                _walk(node.f, acc_x, acc_y)
            return

        # ---- leaf: record accumulated orders ----------------------------
        if acc_x or acc_y:
            out.add((acc_x, acc_y))
            acc_x = acc_y = 0

        # ---- recurse over children --------------------------------------
        for child in node.__dict__.values():
            if isinstance(child, Expression):
                _walk(child, acc_x, acc_y)
            elif isinstance(child, (list, tuple)):
                for c in child:
                    if isinstance(c, Expression):
                        _walk(c, acc_x, acc_y)

    _walk(expr)
    return out