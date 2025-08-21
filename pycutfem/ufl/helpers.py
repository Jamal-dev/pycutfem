import re
from matplotlib.pylab import f
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, Set
from pycutfem.ufl.expressions import Expression, Derivative
from pycutfem.ufl.expressions import (
    VectorFunction, TrialFunction, VectorTrialFunction,
    TestFunction, VectorTestFunction, Restriction
)
from pycutfem.ufl.forms import Form, Equation
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Hessian as UFLHessian, Laplacian as UFLLaplacian


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
        A, B = self.data, other.data
        if A.ndim == 2 and B.ndim == 2:
            # LHS: (k,n) and (k,m) -> (n,m)
            return np.einsum("kn,km->nm", A, B, optimize=True)
        if A.ndim == 1 and B.ndim == 2:
            # RHS: (k,) and (k,n) -> (n,)
            return np.einsum("k,kn->n", A, B, optimize=True)
        if A.ndim == 2 and B.ndim == 1:
            # RHS: (k,n) and (k,) -> (n,)
            return np.einsum("kn,k->n", A, B, optimize=True)
        if A.ndim == 1 and B.ndim == 1:
            # RHS: (k,) and (k,) -> scalar
            return float(np.einsum("k,k->", A, B, optimize=True))
        raise ValueError(f"Unsupported inner dims A{A.shape}, B{B.shape} for VecOpInfo.")
    

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
            raise ValueError(f"Constant vector of size {const.size} is wrong length for VecOpInfo with {self.data.shape} components.")
        #data =  np.einsum("kn,k->n", self.data, const, optimize=True)
        data = self.data.T @ const  # fast BLAS-2
        return VecOpInfo(data[np.newaxis,:], role=self.role)
    def dot_grad(self, grad: "GradOpInfo") -> "VecOpInfo":
        """
        Computes dot(v, ∇u) for a vector basis function v and gradient ∇u.
        Returns a new VecOpInfo with shape (k, n).
        """
        # print(f"VecOpInfo.dot_grad: {self.role} and {grad.role}"
        #           f" with shapes {self.data.shape} and {grad.data.shape}")
        if not isinstance(grad, GradOpInfo):
            raise TypeError(f"Expected GradOpInfo, got {type(grad)}.")
        if self.data.shape[0] != grad.data.shape[0]:
            raise ValueError(f"VecOpInfo {self.shape} and GradOpInfo {grad.shape} must have the same number of components.")
        
        # Case 1: Function · Grad(Trial) or Trial · Grad(Function)
        # 1. value of each velocity component at the current Q-point
        if self.role == "function" and grad.role == "trial":
            # Case:  Function · Grad(Trial)      u_k · ∇u_trial
            # (1)  value of u_k at this quad-point
            u_val = np.sum(self.data, axis=1)          # shape (k,)  —   u_k(ξ)
            # (2)  w_i,n = Σ_d u_d ∂_d φ_{k,n}
            if grad.shape[0] == 1:
                # Special case: single component gradient
                data = np.einsum("d,kld->kl", u_val, grad.data, optimize=True)
            else:
                data = np.einsum("s,sld->dl", u_val, grad.data, optimize=True)
            return VecOpInfo(data, role=grad.role)
        elif self.role == "trial" and grad.role == "function":
            # Case:  Trial · Grad(Function)      u_trial · ∇u_k
            # (1)  value of u_trial at this quad-point
            if grad.coeffs is not None:
                grad_val = np.einsum("knd,kn->kd", grad.data, grad.coeffs, optimize=True)
            else:
                grad_val = grad.data
            # (2)  w_i,n = Σ_d ∂_d φ_{k,n} u_d
            data = np.einsum("sl,sd->dl", self.data, grad_val, optimize=True)
            return VecOpInfo(data, role=self.role)
        elif self.role == "function" and grad.role == "function":
            # Case:  Function · Grad(Function)      u_k · ∇u_k
            # (1)  value of u_k at this quad-point
            u_val = np.sum(self.data, axis=1)          # shape (k,)  —   u_k(ξ)
            if grad.coeffs is not None:
                grad_val = np.einsum("knd,kn->kd", grad.data, grad.coeffs, optimize=True)
            else:
                grad_val = grad.data
            data = np.einsum("s,sd->d", u_val, grad_val, optimize=True)
            return data
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
            elif other.ndim == 1 and self.data.shape[0] == 1:
                # New Case: Scalar multiplication with a vector
                if self.role == "function":
                    vals = np.sum(self.data, axis=1)  # Shape (k,)
                    return vals * other  # returns a 1D array
                return VecOpInfo([self.data[0,:] * comp for comp in other], role=self.role)
            else:
                raise ValueError(f"Cannot multiply VecOpInfo {self.data.shape} with array of shape {other.shape}."
                                 f" Roles: {self.role}, other={getattr(other, 'role', None)}.")
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

    def transpose(self) -> "GradOpInfo":
        """
        Transpose over component and spatial axes.

        (k, n, d) -> (d, n, k)     e.g. [[∂x u1, ∂y u1],
                                        [∂x u2, ∂y u2]]^T
                                    = [[∂x u1, ∂x u2],
                                        [∂y u1, ∂y u2]]

        (k, d)     -> (d, k)
        """
        if self.data.ndim == 3:        # (k, n, d)
            if self.role == "function":
                # Transpose for function gradients
                # swap the coefficents to match the new shape
                if self.coeffs is not None:
                    grad_vals = np.einsum("knd,kn->kd", self.data, self.coeffs, optimize=True)
                return GradOpInfo(grad_vals.T, role=self.role, coeffs=None)
            else: # trial and test 
                return GradOpInfo(self.data.transpose(2, 1, 0), role=self.role, coeffs=self.coeffs)
        if self.data.ndim == 2:        # (k, d) or (n, d)
            return GradOpInfo(self.data.T, role=self.role, coeffs=self.coeffs)
        raise ValueError(f"Cannot transpose GradOpInfo with data of shape {self.data.shape}.")

    def _eval_function_to_2d(self) -> np.ndarray:
        assert self.role == "function", "Only use on function gradients."
        # If it's 3D, contract over n → (k,d)
        if self.data.ndim == 3:
            if self.coeffs is not None:
                # (k, n, d) contracted with (k, n) → (k, d)
                kd = np.einsum("knd,kn->kd", self.data, self.coeffs, optimize=True)
                return kd
            else:
                raise ValueError("Function gradient is 3D but no coeffs provided.")

        # Already 2D: try to use a stored hint, otherwise assume ('k','d')
        if self.data.ndim == 2:
            return self.data

        raise ValueError(f"Unexpected function gradient shape: {self.data.shape}")

    def inner(self, other: "GradOpInfo") -> np.ndarray:
        if not isinstance(other, GradOpInfo) or self.data.shape != other.data.shape:
            raise ValueError(...)

        if self.role == "test" and other.role == "trial":
            # standard order: rows=test, cols=trial
            return np.einsum("knd,kmd->nm", self.data, other.data, optimize=True)

        elif self.role == "trial" and other.role == "test":
            # reversed inputs: build rows=test, cols=trial anyway
            return np.einsum("knd,kmd->mn", self.data, other.data, optimize=True)

        elif self.role == "function" and other.role == "function":
            # (RHS or unusual cases rarely hit here; keep the default if needed)
            return np.einsum("knd,kmd->nm", self.data, other.data, optimize=True)
        else:
            raise NotImplementedError(f"GradOpInfo.inner not implemented for roles {self.role} and {other.role}."
                                      f" Shapes: {self.data.shape} and {other.data.shape}."
                                      f" Roles: {self.role} and {other.role}.")


    def dot(self, other:"GradOpInfo") -> "GradOpInfo":
        """
        Computes dot(∇u, ∇v) for two GradOpInfo objects.
        Returns a new GradOpInfo with shape (k, n, d).
        """
        if not isinstance(other, GradOpInfo):
            raise TypeError(f"Expected GradOpInfo, got {type(other)}.")
        if self.data.shape[-1] != other.data.shape[0]:
            raise ValueError(f"GradOpInfo shapes mismatch: {self.data.shape} vs {other.data.shape}.")
        
        # Case 1: Function · Grad(Trial) or Trial · Grad(Function)
        if self.role == "function" and other.role == "trial":
            # Case:  Function · Grad(Trial)      ∇u_k · ∇u_trial
            # (k, d)   =  Σ_i  u_{k,i}  ∂_d φ_i   – true ∇u_k at this quad-point
            if self.coeffs is not None:
                grad_val = np.einsum("knd,kn->kd", self.data, self.coeffs, optimize=True)
            else: 
                grad_val = self.data
            # (2)  matrix multiplication
            data = np.einsum("ks,snd->knd", grad_val, other.data, optimize=True)
            return GradOpInfo(data, role=other.role)
        elif self.role == "trial" and other.role == "function":
            # Case:  Grad(Trial) · Grad(Function)      ∇u_trial · ∇u_k
            # (1)  value of u_trial at this quad-point
            if other.coeffs is not None:
                grad_val = np.einsum("knd,kn->kd", other.data, other.coeffs, optimize=True)
            else:
                grad_val = other.data
            data = np.einsum("kns,sd->knd", self.data, grad_val, optimize=True)
            return GradOpInfo(data, role=self.role)
        elif self.role == "function" and other.role == "function":
            # Case:  Grad(Function) · Grad(Function)      ∇u_k · ∇u_k
            # (1)  value of ∇u_k at this quad-point
            if self.coeffs is not None:
                grad_val = np.einsum("knd,kn->kd", self.data, self.coeffs, optimize=True)
            else:
                grad_val = self.data
            if other.coeffs is not None:
                other_grad_val = np.einsum("knd,kn->kd", other.data, other.coeffs, optimize=True)
            else:
                other_grad_val = other.data
            # Matrix multiplication
            data = np.einsum("kd,dn->kn", grad_val, other_grad_val, optimize=True)
            return GradOpInfo(data, role=self.role)
        else:
            raise NotImplementedError(f"GradOpInfo.dot not implemented for roles {self.role} and {other.role}."
                                      f" Shapes: {self.data.shape} and {other.data.shape}.")

    def left_dot(self, vec:np.ndarray) -> VecOpInfo:
        """
        Computes the left dot product with a constant vector or VecOpInfo over the SPATIAL dimension.
        This operation reduces the dimension and returns a VecOpInfo object.
        dot(c, ∇v) -> c ⋅ ∇v
        """
        if isinstance(vec, (VecOpInfo)):
            # Case:  Const · Grad(Function)      c · ∇u_k
            if self.role == "function":
                # (1)  value of ∇u_k at this quad-point
                if self.coeffs is not None:
                    grad_val = np.einsum("knd,kn->kd", self.data, self.coeffs, optimize=True)
                else:
                    grad_val = self.data
                # (2)  dot product
                if vec.role in {"trial", "test"}:
                    data = np.einsum("kn,kd->dn", vec.data, grad_val, optimize=True)
                    return VecOpInfo(data, role=vec.role)
                elif vec.role == "function":
                    u_vals = np.sum(vec.data, axis=1)  # Shape (k,)
                    return np.dot(u_vals, grad_val)  # scalar result for rhs
            elif self.role in {"trial", "test"}:
                if vec.role == "function":
                    u_vals = np.sum(vec.data, axis=1)  # Shape (k,)
                    if self.data.shape[0] == 1:
                        # Special case: single component gradient
                        data = np.einsum("d,kld->kl", u_vals, self.data, optimize=True)
                    else:
                        data = np.einsum("s,snd->dn", u_vals, self.data, optimize=True)
                    return VecOpInfo(data, role=self.role)

        if isinstance(vec, np.ndarray) and vec.ndim == 1:
            if self.role == "function":
                # Case:  Const · Grad(Function)      c · ∇u_k
                if self.coeffs is not None:
                    grad_val = np.einsum("knd,kn->kd", self.data, self.coeffs, optimize=True)
                else:
                    grad_val = self.data
                if grad_val.shape[0] == 1:
                    # Special case: single component gradient
                    data = np.einsum("d,kd->k", vec, grad_val, optimize=True)
                    return data # returns a 1D array
                else:
                    data = np.einsum("k,kd->d", vec, grad_val, optimize=True)
                return data
            elif self.role in {"trial", "test"}:
                if self.data.shape[0] == 1:
                    # Special case: single component gradient
                    data = np.einsum("d,kld->kl", vec, self.data, optimize=True)
                else:
                    data = np.einsum("s,snd->dn", vec, self.data, optimize=True)
                return VecOpInfo(data, role=self.role)

        raise ValueError(f"Cannot left_dot with vector of shape {vec.shape}.")

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
                if self.coeffs is not None:
                    grad_val = np.einsum("knd,kn->kd", self.data, self.coeffs, optimize=True)
                else:
                    grad_val = self.data

                # (k, n)   =  Σ_d  (∇u_k)_d  φ_{u,d,n}     – (∇u_k)·δu  for every trial dof n
                # data = grad_val @ vec.data                  # BLAS-2, same as einsum("kd,dn->kn")
                data = grad_val @ vec.data
                
                return VecOpInfo(data, role=vec.role)
            
            if self.role == "trial" and vec.role == "function": # introducing a new branch
                # Case:  Grad(Trial) · Vec(Function)      (∇u_trial) · u_k
                v_val = np.sum(vec.data, axis=1)          # (k,d) ∫ u_k dx ) 
                data = np.einsum("kld,d->kl", self.data, v_val, optimize=True)
                return VecOpInfo(data, role=self.role)
            if self.role == "function" and vec.role == "function":
                # Case:  Grad(Function) · Vec(Function)      (∇u_k) · u_k
                # (1)  value of ∇u_k at this quad-point
                if self.coeffs is not None:
                    grad_val = np.einsum("knd,kn->kd", self.data, self.coeffs, optimize=True)
                else:
                    grad_val = self.data
                v_val = np.sum(vec.data, axis=1)
                w_val = np.einsum("kd,d->k", grad_val, v_val, optimize=True)
                return w_val # rhs, so return a 1D array

            
        
        if isinstance(vec, np.ndarray) and vec.ndim == 1: # dot product with a constant vector
            # print(f"GradOpInfo.dot_vec: vec={vec}, data.shape={self.data.shape}, role={self.role}"
                #   f", vec.role={getattr(vec, 'role', None)}, vec.shape={getattr(vec, 'shape', None)}")
            if self.role == "function":
                # Case:  Grad(Function) · Const      (∇u_k) · c
                # (1)  value of ∇u_k at this quad-point
                if self.coeffs is not None and self.data.shape != (2,2):
                    # (k, d)   =  Σ_i  u_{k,i}  ∂_d φ_i   – true ∇u_k at this quad-point
                    grad_val = np.einsum("knd,kn->kd", self.data, self.coeffs, optimize=True)
                else:
                    grad_val = self.data
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
        if not isinstance(other, GradOpInfo) and self.role != other.role:
            raise ValueError("Operands must be GradOpInfo of the same shape for addition."
                             f" Shapes: {self.data.shape} and {other.data.shape}."
                             f" Roles: {self.role} and {other.role}.")
        if self.role in ["test", "trial"] and other.role in ["test", "trial"]:
            # Case: both are test or trial gradients
            if self.data.shape != other.data.shape:
                raise ValueError(f"GradOpInfo shapes mismatch in addition: {self.data.shape} vs {other.data.shape}.")
            return GradOpInfo(self.data + other.data, role=self.role)
        elif self.role == "function" and other.role == "function":
            a = self._eval_function_to_2d()  
            b = other._eval_function_to_2d()
            if a.shape != b.shape:
                raise ValueError(f"Function gradient shapes mismatch in addition: {a.shape} vs {b.shape}.")
            return GradOpInfo(a + b, role=self.role) # collapsed gradients (2,2)
        else:
            raise NotImplementedError(f"GradOpInfo addition not implemented for roles {self.role} and {other.role}."
                                      f" Shapes: {self.data.shape} and {other.data.shape}.")

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

def make_hessian_from_ref(d20, d11, d02, role: str) -> "HessOpInfo":
    """
    d20,d11,d02 already padded to union length and stacked per component:
    each of shape (k, n). Return HessOpInfo (k,n,2,2).
    """
    k, n = d20.shape
    H = np.empty((k, n, 2, 2), dtype=d20.dtype)
    H[..., 0, 0] = d20
    H[..., 0, 1] = d11
    H[..., 1, 0] = d11
    H[..., 1, 1] = d02
    return HessOpInfo(H, role=role)

def make_laplacian_from_ref(d20, d02, role: str) -> VecOpInfo:
    return VecOpInfo(d20 + d02, role=role)   # (k, n)




@dataclass(slots=True, frozen=True)
class HessOpInfo:
    """
    Hessian of basis functions, per component:
    data shape: (k, n, d, d) — k components, n (union) dofs, d=2 spatial.
    role: "test", "trial", or "function"
    """
    data: np.ndarray
    role: str = "none"
    coeffs: np.ndarray = field(default=None)

    # ---------- linear algebra ops on the Hessian tensor -----------------

    def transpose(self) -> "HessOpInfo":
        """Swap the last two (spatial) axes i↔j; keep (k, n) intact."""
        return HessOpInfo(self.data.swapaxes(-1, -2), role=self.role, coeffs=self.coeffs)

    def __neg__(self) -> "HessOpInfo":
        return HessOpInfo(-self.data, role=self.role, coeffs=self.coeffs)

    def __mul__(self, other):
        """Scalar multiplication (left or right)."""
        if isinstance(other, (int, float, np.floating)):
            return HessOpInfo(self.data * other, role=self.role, coeffs=self.coeffs)
        if isinstance(other, np.ndarray) and other.ndim == 0:
            return HessOpInfo(self.data * float(other), role=self.role, coeffs=self.coeffs)
        raise TypeError(f"HessOpInfo can only be multiplied by scalars, not {type(other)}")
    __rmul__ = __mul__

    def __add__(self, other: "HessOpInfo") -> "HessOpInfo":
        if not isinstance(other, HessOpInfo):
            raise TypeError(f"Cannot add HessOpInfo to {type(other)}.")
        if self.data.shape != other.data.shape:
            raise ValueError(f"HessOpInfo shapes mismatch in addition: {self.data.shape} vs {other.data.shape}.")
        if self.role != other.role:
            raise ValueError(f"Cannot add HessOpInfo with different roles: {self.role} vs {other.role}.")
        # prefer keeping coeffs only if both match, otherwise drop
        coeffs = self.coeffs if (self.coeffs is other.coeffs) else None

        return HessOpInfo(self.data + other.data, role=self.role, coeffs=coeffs)

    def __sub__(self, other: "HessOpInfo") -> "HessOpInfo":
        return self.__add__(-other)

    # ---------- algebra used in assembly ---------------------------------
    def trace(self) -> "VecOpInfo":
        """
        Trace over spatial axes → per‑component Laplacian table with shape (k, n).
        Note: for role=='function' we purposely *do not* contract with coeffs here.
        That contraction (if needed) is handled at the integrand level (e.g. in _visit_Inner).
        """
        tr = self.data[..., 0, 0] + self.data[..., 1, 1]      # (k, n)
        return VecOpInfo(tr, role=self.role)



    # ---------- vector contractions --------------------------------------
    def dot_right(self, vec) -> "GradOpInfo":
        """Right contraction H · n over the last spatial axis j.
        Returns:
          test/trial → GradOpInfo with data (k,n,d)
          function   → GradOpInfo with data (k,d)
        """
        role_vec = vec.role if hasattr(vec, "role") else None
        n = np.asarray(vec.data if hasattr(vec, "data") else vec)
        if role_vec == "function" and n.ndim == 2:
            # collapse the axis
            n = np.sum(vec.data, axis=1)  # Shape (k,)
        
        if self.data.ndim == 4:    # (k,n,d,d) · (d,) -> (k,n,d)
            if self.data.shape[0] == 1:
                # special case for scalar
                out = np.einsum("knij,j->kni", self.data, n, optimize=True)
            else:
                out = np.einsum("knij,j->kni", self.data, n, optimize=True)
        elif self.data.ndim == 3:  # (k,d,d)   · (d,) -> (k,d)
            out = np.einsum("kij,j->ki",   self.data, n, optimize=True)
        else:
            raise ValueError(f"HessOpInfo.dot_right: unexpected ndim={self.data.ndim}")
        return GradOpInfo(out, role=self.role)

    def dot_left(self, vec) -> "GradOpInfo":
        """Left contraction n · H over the first spatial axis i."""
        role_vec = vec.role if hasattr(vec, "role") else None
        n = np.asarray(vec.data if hasattr(vec, "data") else vec)

        if role_vec == "function" and n.ndim == 2:
            # collapse the axis
            n = np.sum(vec.data, axis=1)  # Shape (k,)

        if self.data.ndim == 4:    # (d,) · (k,n,d,d) -> (k,n,d)
            if self.data.shape[0] == 1:
                out = np.einsum("s,knsj->knj", n, self.data, optimize=True)
            else:
                out = np.einsum("s,snij->inj", n, self.data, optimize=True)
        elif self.data.ndim == 3:  # (d,) · (k,d,d)   -> (k,d)
            out = np.einsum("s,sij->ij",   n, self.data, optimize=True)
        else:
            raise ValueError(f"HessOpInfo.dot_left: unexpected ndim={self.data.ndim}")
        return GradOpInfo(out, role=self.role)

    def proj_nn(self, nvec) -> "VecOpInfo":
        """Double contraction nᵀ H n (second normal derivative).
        Returns:
          test/trial → VecOpInfo with data (k,n)
          function   → VecOpInfo with data (k,)
        """
        n = np.asarray(nvec.data if hasattr(nvec, "data") else nvec).reshape(-1)
        if self.data.ndim == 4:    # (d,) · (k,n,d,d) · (d,) -> (k,n)
            tmp = np.einsum("s,snij->inj", n, self.data, optimize=True)
            val = np.einsum("knj,j->kn",   tmp, n, optimize=True)
        elif self.data.ndim == 3:  # (d,) · (k,d,d)   · (d,) -> (k,)
            tmp = np.einsum("s,sij->ij",   n, self.data, optimize=True)
            val = np.einsum("kj,j->k",     tmp, n, optimize=True)
        else:
            raise ValueError(f"HessOpInfo.proj_nn: unexpected ndim={self.data.ndim}")
        return VecOpInfo(val, role=self.role)

    def inner(self, other: "HessOpInfo") -> np.ndarray:
        """
        Frobenius inner product per DOF pair:

          (n_test, n_trial) ← einsum("k n i j, k m i j -> n m", Htest, Htrial)
        """
        A, B = self.data, other.data
        if A.ndim != 4 or B.ndim != 4:
            raise ValueError("HessOpInfo.inner expects (k,n,2,2) arrays.")
        if A.shape[0] != B.shape[0] or A.shape[2:] != B.shape[2:]:
            raise ValueError(f"HessOpInfo component/spatial mismatch: {A.shape} vs {B.shape}")
        return np.einsum("knij,kmij->nm", A, B, optimize=True)



    @property
    def shape(self): return self.data.shape
    @property
    def ndim(self): return self.data.ndim
    def __repr__(self): return f"HessOpInfo(shape={self.data.shape}, role='{self.role}')"

    def info(self): return f"HessOpInfo({self.data.dtype}, shape={self.data.shape}, role='{self.role}')"



# ========================================================================
# Generic walker
def _iter_child_exprs(node):
    for value in node.__dict__.values():
        if isinstance(value, Expression):
            yield value
        elif isinstance(value, (list, tuple)):
            for v in value:
                if isinstance(v, Expression):
                    yield v

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
        if isinstance(node, UFLHessian):
            out.update({(2,0),(1,1),(0,2)}); _walk(node.operand, acc_x, acc_y); return
        if isinstance(node, UFLLaplacian):
            out.update({(2,0),(0,2)}); _walk(node.operand, acc_x, acc_y); return

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


# ------------------------------------------------------------------
#  all_fields -------------------------------------------------------
# ------------------------------------------------------------------
def _all_fields(expr):
    fields   = set()
    visited  = set()

    def walk(n):
        nid = id(n)
        if nid in visited:
            return
        visited.add(nid)

        # record
        if getattr(n, "field_name", None):
            fields.add(n.field_name)
        if getattr(n, "space", None):
            fields.update(n.space.field_names)
        if isinstance(n, VectorFunction):
            fields.update(n.field_names)

        # recurse
        for child in _iter_child_exprs(n):
            walk(child)

    walk(expr)
    return list(fields)

# ------------------------------------------------------------------
#  find_all ---------------------------------------------------------
# ------------------------------------------------------------------
def _find_all(expr, cls):
    out      = []
    visited  = set()

    def walk(n):
        nid = id(n)
        if nid in visited:
            return
        visited.add(nid)

        if isinstance(n, cls):
            out.append(n)
        for child in _iter_child_exprs(n):
            walk(child)

    walk(expr)
    return out

# New: Helper to identify trial and test functions in an expression
def _trial_test(expr): 
    """Finds the first trial and test function in an expression tree.""" 
    trial = expr.find_first(lambda n: isinstance(n, (TrialFunction, VectorTrialFunction))) 
    test = expr.find_first(lambda n: isinstance(n, (TestFunction, VectorTestFunction))) 
    return trial, test

# ------------------------------------------------------------------
#  find_all_restrictions -------------------------------------------
# ------------------------------------------------------------------
def _find_all_restrictions(form) -> list[Restriction]:
    restrictions = []
    visited      = set()

    def walk(obj):
        oid = id(obj)
        if oid in visited:
            return
        visited.add(oid)

        if isinstance(obj, Restriction):
            restrictions.append(obj)

        if isinstance(obj, Expression):
            iterable = _iter_child_exprs(obj)
        else:  # Form / Integral containers
            iterable = []
            for v in getattr(obj, "__dict__", {}).values():
                if isinstance(v, Expression):
                    iterable.append(v)
                elif isinstance(v, (list, tuple)):
                    iterable.extend([c for c in v if isinstance(c, Expression)])

        for child in iterable:
            walk(child)

    walk(form)
    return restrictions


def analyze_active_dofs(equation: Equation, dh: DofHandler, me: MixedElement, bcs: list):
    """
    Return (active_dofs, has_restriction).
    - active_dofs: indices touched by Restricted domains if any, otherwise all DOFs.
    - has_restriction: True iff Restriction operators are present in the forms.
    """
    active_dof_set = set()
    all_forms = equation.a.integrals + equation.L.integrals
    all_restrictions = _find_all_restrictions(Form(all_forms))

    if all_restrictions:
        print("Restriction operators found. Analyzing active domains...")
        for r in all_restrictions:
            fields_in_operand = _all_fields(r.operand)
            active_element_ids = r.domain.to_indices()
            for eid in active_element_ids:
                elemental_dofs_vector = dh.get_elemental_dofs(eid)
                for field_name in fields_in_operand:
                    sl = me.component_dof_slices[field_name]
                    active_dof_set.update(elemental_dofs_vector[sl])
        return np.array(sorted(active_dof_set), dtype=int), True
    else:
        print("No Restriction operators found. All DOFs are considered active.")
        return np.arange(dh.total_dofs, dtype=int), False

from pycutfem.fem.transform import JET_CACHE, RefDerivCache
from pycutfem.fem import transform



def phys_scalar_third_row(me, fld: str, xi: float, eta: float,
                          i: int, j: int, k: int, mesh, elem_id: int,
                          ref_cache: RefDerivCache | None = None) -> np.ndarray:
    """
    Build the (n_loc,) row for the 3rd-order physical derivative ∂^3/∂x^i ∂x^j ∂x^k of scalar field 'fld'
    at the reference point (xi,eta) in element elem_id.

      f_{ijk} = g_{IJK} A^I_i A^J_j A^K_k
              + g_{IJ} ( A^I_{ij} A^J_k + A^I_{ik} A^J_j + A^I_{jk} A^J_i )
              + g_I A^I_{ijk}

    where g_* are *reference* derivatives of the basis functions.
    Cached row for f_{ijk} at (xi,eta).
    """
    if ref_cache is None:
        ref_cache = RefDerivCache(me)

    rec = JET_CACHE.get(mesh, elem_id, xi, eta, upto=3)
    A  = rec["A"]; A2 = rec["A2"]; A3 = rec["A3"]

    # reference derivatives (cached)
    g1 = [ref_cache.get(fld, xi, eta, 1, 0),
          ref_cache.get(fld, xi, eta, 0, 1)]

    g2 = [[ref_cache.get(fld, xi, eta, 2, 0),
           ref_cache.get(fld, xi, eta, 1, 1)],
          [ref_cache.get(fld, xi, eta, 1, 1),
           ref_cache.get(fld, xi, eta, 0, 2)]]

    g3_300 = ref_cache.get(fld, xi, eta, 3, 0)
    g3_210 = ref_cache.get(fld, xi, eta, 2, 1)
    g3_120 = ref_cache.get(fld, xi, eta, 1, 2)
    g3_030 = ref_cache.get(fld, xi, eta, 0, 3)

    def g3(I, J, K):
        s = I + J + K
        if   s == 0: return g3_300
        elif s == 1: return g3_210
        elif s == 2: return g3_120
        else:        return g3_030

    nloc = g1[0].shape[0]
    row  = np.zeros(nloc, float)

    # g_{IJK} A^I_i A^J_j A^K_k
    for I0 in (0, 1):
        for J0 in (0, 1):
            for K0 in (0, 1):
                row += g3(I0, J0, K0) * A[I0, i] * A[J0, j] * A[K0, k]

    # g_{IJ} (A^I_{ij} A^J_k + A^I_{ik} A^J_j + A^I_{jk} A^J_i)
    for I0 in (0, 1):
        for J0 in (0, 1):
            row += g2[I0][J0] * ( A2[I0, i, j] * A[J0, k]
                                 + A2[I0, i, k] * A[J0, j]
                                 + A2[I0, j, k] * A[J0, i] )

    # g_I A^I_{ijk}
    for I0 in (0, 1):
        row += g1[I0] * A3[I0, i, j, k]

    return row




def phys_scalar_fourth_row(me, fld: str, xi: float, eta: float,
                           i: int, j: int, k: int, l: int, mesh, elem_id: int,
                           ref_cache: RefDerivCache | None = None) -> np.ndarray:
    """
    Row (length n_loc) for  ∂^4/∂x^i ∂x^j ∂x^k ∂x^l  of scalar field 'fld' at (xi,eta), element elem_id.

    f_{ijkl} = g_{IJKL} A^I_i A^J_j A^K_k A^L_l
             + g_{IJK}  sum_{2+1+1} (...)
             + g_{IJ}   ( A^I_{ij}A^J_{kl} + A^I_{ik}A^J_{jl} + A^I_{il}A^J_{jk}
                         + A^I_{ijk}A^J_l + A^I_{ijl}A^J_k + A^I_{ikl}A^J_j + A^I_{jkl}A^J_i )
             + g_I      A^I_{ijkl}.
    """
    if ref_cache is None:
        ref_cache = RefDerivCache(me)

    rec = JET_CACHE.get(mesh, elem_id, xi, eta, upto=4)
    A  = rec["A"]; A2 = rec["A2"]; A3 = rec["A3"]; A4 = rec["A4"]

    # reference derivatives (cached)
    g1 = [ref_cache.get(fld, xi, eta, 1, 0),
          ref_cache.get(fld, xi, eta, 0, 1)]

    g2 = [[ref_cache.get(fld, xi, eta, 2, 0),
           ref_cache.get(fld, xi, eta, 1, 1)],
          [ref_cache.get(fld, xi, eta, 1, 1),
           ref_cache.get(fld, xi, eta, 0, 2)]]

    g3_300 = ref_cache.get(fld, xi, eta, 3, 0)
    g3_210 = ref_cache.get(fld, xi, eta, 2, 1)
    g3_120 = ref_cache.get(fld, xi, eta, 1, 2)
    g3_030 = ref_cache.get(fld, xi, eta, 0, 3)

    g4_400 = ref_cache.get(fld, xi, eta, 4, 0)
    g4_310 = ref_cache.get(fld, xi, eta, 3, 1)
    g4_220 = ref_cache.get(fld, xi, eta, 2, 2)
    g4_130 = ref_cache.get(fld, xi, eta, 1, 3)
    g4_040 = ref_cache.get(fld, xi, eta, 0, 4)

    def g3(I, J, K):
        s = I + J + K
        if   s == 0: return g3_300
        elif s == 1: return g3_210
        elif s == 2: return g3_120
        else:        return g3_030

    def g4(I, J, K, L):
        s = I + J + K + L
        if   s == 0: return g4_400
        elif s == 1: return g4_310
        elif s == 2: return g4_220
        elif s == 3: return g4_130
        else:        return g4_040

    nloc = g1[0].shape[0]
    row  = np.zeros(nloc, float)

    # 1) g_{IJKL} A^I_i A^J_j A^K_k A^L_l
    for I0 in (0, 1):
        for J0 in (0, 1):
            for K0 in (0, 1):
                for L0 in (0, 1):
                    row += g4(I0, J0, K0, L0) * A[I0, i] * A[J0, j] * A[K0, k] * A[L0, l]

    # 2) g_{IJK} (six 2+1+1 terms)
    for I0 in (0, 1):
        for J0 in (0, 1):
            for K0 in (0, 1):
                G3 = g3(I0, J0, K0)
                row += G3 * ( A2[I0, i, j] * A[J0, k] * A[K0, l]
                            + A2[I0, i, k] * A[J0, j] * A[K0, l]
                            + A2[I0, i, l] * A[J0, j] * A[K0, k]
                            + A2[I0, j, k] * A[J0, i] * A[K0, l]
                            + A2[I0, j, l] * A[J0, i] * A[K0, k]
                            + A2[I0, k, l] * A[J0, i] * A[K0, j] )

    # 3) g_{IJ} (pair+pair and 3+1 terms)
    for I0 in (0, 1):
        for J0 in (0, 1):
            G2 = g2[I0][J0]
            row += G2 * ( A2[I0, i, j] * A2[J0, k, l]
                        + A2[I0, i, k] * A2[J0, j, l]
                        + A2[I0, i, l] * A2[J0, j, k] )
            row += G2 * ( A3[I0, i, j, k] * A[J0, l]
                        + A3[I0, i, j, l] * A[J0, k]
                        + A3[I0, i, k, l] * A[J0, j]
                        + A3[I0, j, k, l] * A[J0, i] )

    # 4) g_I A^I_{ijkl}
    for I0 in (0, 1):
        row += g1[I0] * A4[I0, i, j, k, l]

    return row

def _as_indices(ox: int, oy: int) -> tuple[int, ...]:
    """
    Map mixed-partial counts (ox, oy) to a canonical index tuple in {0,1}^k,
    where 0 := x, 1 := y and k = ox + oy.

    Examples
    --------
    order 3:
      (3,0) -> (0,0,0)
      (2,1) -> (0,0,1)
      (1,2) -> (0,1,1)
      (0,3) -> (1,1,1)

    order 4:
      (4,0) -> (0,0,0,0)
      (3,1) -> (0,0,0,1)
      (2,2) -> (0,0,1,1)
      (1,3) -> (0,1,1,1)
      (0,4) -> (1,1,1,1)

    Notes
    -----
    - The high-order formulas for f_{ijk} and f_{ijkl} are symmetric in the
      spatial indices, so this canonical ordering is sufficient.
    - If you ever need a different permutation, you can reorder the returned
      tuple before passing it on.
    """
    if ox < 0 or oy < 0:
        raise ValueError("ox, oy must be nonnegative")
    return tuple([0] * ox + [1] * oy)
