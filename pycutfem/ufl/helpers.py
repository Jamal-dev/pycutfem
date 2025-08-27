import re
from matplotlib.pylab import f
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, Set, Sequence, Optional, Dict, Any
from pycutfem.ufl.expressions import Expression, Derivative
from pycutfem.ufl.expressions import (
    VectorFunction, TrialFunction, VectorTrialFunction,
    TestFunction, VectorTestFunction, Restriction
)
from pycutfem.ufl.forms import Form, Equation
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Hessian as UFLHessian, Laplacian as UFLLaplacian
from pycutfem.ufl.helpers_geom import phi_eval


import logging
# Setup logging
logger = logging.getLogger(__name__)
def _meta_has(val, kind, member_names = ("field_names", "parent_name", "side", "field_sides", "is_rhs")):
    for name in member_names:
        if name == kind: return bool(val)
    return False
def _resolve_meta(a, b, prefer=None, strict=False):
    """
    Merge metadata dicts {'field_names','parent_name','side','field_sides'}.
    prefer ∈ {None,'a','b'}; strict -> raise on field_names/parent_name conflict.
    """
    def pick(attr, A, B):
        ah, bh = _meta_has(A, attr), _meta_has(B, attr)
        if prefer == "a" and ah: return A
        if prefer == "b" and bh: return B
        if ah and not bh: return A
        if bh and not ah: return B
        if ah and bh:
            if A == B: return A
            if strict and attr in ("field_names","parent_name"):
                raise ValueError(f"Metadata conflict for '{attr}': {A} vs {B}")
            # degrade on conflict
            return [] if attr in ("field_names","field_sides") else ""
        # default (empty)
        return [] if attr in ("field_names","field_sides") else ""
    return {
        k: pick(k, a.get(k, [] if k in ("field_names","field_sides") else ""),
                   b.get(k, [] if k in ("field_names","field_sides") else ""))
        for k in ("field_names","parent_name","side","field_sides", "is_rhs")
    }
def _collapsed_grad(grad) -> np.ndarray:
    """Collapse GradOpInfo to values at this QP: returns (k_u, d)."""
    if grad.coeffs is not None and grad.data.ndim == 3:
        grad_val = np.einsum("knd,kn->kd", grad.data, grad.coeffs, optimize=True)
    else:
        grad_val = grad.data
    return grad_val # (k, d)
def _collapsed_function(func) -> np.ndarray:
    """Collapse VecOpInfo(function) to values at this QP: returns (k,)."""
    if func.data.ndim == 2:
        func_val = np.sum(func.data, axis=1)
    else:
        func_val = func.data
    return func_val # (k,)
def _is_1d_vector(vec) -> bool:
    """Check if the given vector is 1D."""
    return vec.ndim == 1 and vec.shape[0] > 1
def _is_scalar(a) -> bool:
    """Check if the given vector is a scalar."""
    return np.isscalar(a)

@dataclass(frozen=True, slots=True)
class BaseOpInfo:
    """Base class for operation information containers."""
    data: np.ndarray
    role: str = "none"
    field_names: list = field(default_factory=list)
    parent_name: str = ""
    side: str = ""                   # '+', '-', or ''
    field_sides: list = field(default_factory=list)
    is_rhs: bool = True
    def meta(self):
        return {
            "field_names": self.field_names,
            "parent_name": self.parent_name,
            "side": self.side,
            "field_sides": self.field_sides,
            "is_rhs": self.is_rhs
        }
    def update_meta(self, meta: dict) -> dict:
        # tolerate callers that merge only a subset of keys
        return {
            k: meta.get(k, getattr(self, k))
            for k in ("field_names", "parent_name", "side", "field_sides", "is_rhs")
        }
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
    def _error_msg(self, other, opname: str) -> str:
        """Auto-formats a helpful error; no need to override in subclasses."""
        cls_a = self.__class__.__name__
        if isinstance(other, BaseOpInfo):
            cls_b = other.__class__.__name__
            role_b = other.role
            shape_b = other.shape
            fields_b = other.field_names
            side_b = other.side
            parent_b = other.parent_name
            field_sides_b = other.field_sides
            is_rhs_b = other.is_rhs
        elif isinstance(other, np.ndarray):
            cls_b = "ndarray"
            role_b = None
            shape_b = other.shape
            fields_b = []
            side_b = None
            parent_b = None
            field_sides_b = []
            is_rhs_b = None
        else:
            cls_b = other.__class__.__name__
            role_b = getattr(other, 'role', None)
            shape_b = getattr(other, 'shape', None)
            fields_b = getattr(other, 'field_names', [])
            side_b = getattr(other, 'side', None)
            parent_b = getattr(other, 'parent_name', None)
            field_sides_b = getattr(other, 'field_sides', [])
            is_rhs_b = getattr(other, 'is_rhs', None)
        return (f"Cannot perform {opname} between {cls_a}(role={self.role}, shape={self.shape}, "
                f"fields={self.field_names}, side={self.side}, parent={self.parent_name}, "
                f"field_sides={self.field_sides}, is_rhs={self.is_rhs}) and "
                f"{cls_b}(role={role_b}, shape={shape_b}, fields={fields_b}, side={side_b}, "
                f"parent={parent_b}, field_sides={field_sides_b}, is_rhs={is_rhs_b})")
# ========================================================================
#  Tensor Containers for Symbolic Basis Functions
# ========================================================================
@dataclass(slots=True, frozen=True)
class VecOpInfo(BaseOpInfo):
    """Container for vector basis functions φ (phi). Shape: (k, n)."""
    # data shape: (num_components=k, num_local_dofs=n)
    def _with(self, data: np.ndarray, role: str | None = None) -> "VecOpInfo":
        return VecOpInfo(data, role=(role or self.role),
                        field_names=self.field_names,
                        parent_name=self.parent_name, side=self.side,
                        field_sides=self.field_sides, is_rhs=self.is_rhs)
    
    def _expand_axis_lhs(self, data: np.ndarray, role: str, meta: dict):
        if self.is_rhs: 
            return data
        else: # lhs
            return VecOpInfo(data[np.newaxis,:], role=role, **self.update_meta(meta)) # (1,n)
    

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
    

    def dot_const(self, const: np.ndarray):
        """Computes dot(v, c), returning an (n,) vector."""
        # case for Constant with dim = 1 and also normal vector
        logger.debug(f"VecOpInfo.dot_const: const={const}, data.shape={self.data.shape}")
        const = np.asarray(const)
        if self.role == "function":
            vals = _collapsed_function(self.data)
            if const.shape[0] == vals.shape[0]:
                data = np.dot(vals, const)  # scalar result
                return self._with(data, role="scalar")
            elif vals.shape[0] == 1 and _is_1d_vector(const): # scalar function dot with vector
                # make it to vector function by broadcasting
                data = vals[0] * const # (k_vec,)
                return self._with(data, role=self.role)
        elif self.role in {"trial","test"}:
            if self.data.shape[0] == 1 and _is_1d_vector(const): # scalar basis dot with vector
                data = np.asarray([self.data[0,:] * comp for comp in const])
                return self._with(data, role=self.role)
            elif self.data.shape[0] == const.shape[0]:
                data = np.einsum("kn,k->n", self.data, const, optimize=True)
                if self.is_rhs:
                    return data
                return self._with(data[np.newaxis,:], role=self.role) # scalar basis on lhs
        elif self.role == "scalar":
            if _is_1d_vector(const):
                role = "vector"
            else: role = "scalar"
            return self._with(self.data * const, role=role)
        elif self.role == "vector":
            if _is_1d_vector(const):
                role = "scalar"
                return self._with(np.dot(self.data, const), role=role)
            else: 
                role = "vector"
                return self._with(self.data * const, role=role)
        else:
            raise NotImplementedError(self._error_msg(const, "dot_const"))

    def dot_grad(self, grad: "GradOpInfo") -> "VecOpInfo":
        """
        Computes dot(v, ∇u) for a vector basis function v and gradient ∇u.
        Returns a new VecOpInfo with shape (k_u, n), where k_u is the number
        of components of the operand of the gradient (scalar → 1).
        """
        # print(f"VecOpInfo.dot_grad: {self.role} and {grad.role}"
        #           f" with shapes {self.data.shape} and {grad.data.shape}")
        if not isinstance(grad, GradOpInfo):
            raise TypeError(f"Expected GradOpInfo, got {type(grad)}.")
        meta_grad = grad.meta() if hasattr(grad, "meta") else {
            "field_names": [], "parent_name": "", "side": "", "field_sides": [], "is_rhs": self.is_rhs
        }
        
        # Case 1: Function · Grad(Trial)   -> (k_u, n)
        if self.role == "function" and grad.role == "trial":
            # Case:  Function · Grad(Trial)      u_k · ∇u_trial
            # (1)  value of u_k at this quad-point
            u_val = _collapsed_function(self)  # shape (k,)  —   u_k(ξ)
            if u_val.shape[0] > 1:
                if grad.shape[0] == 1:
                    # Special case: single component gradient
                    data = np.einsum("d,kld->kl", u_val, grad.data, optimize=True)
                else:
                    data = np.einsum("s,sld->dl", u_val, grad.data, optimize=True)
                meta = _resolve_meta(self.meta(), meta_grad, prefer='b')
                return VecOpInfo(data, role=grad.role, **self.update_meta(meta))
            elif u_val.shape[0] == 1:
                meta = _resolve_meta(self.meta(), meta_grad, prefer='b')
                if grad.shape[0] == 1:
                    data = np.einsum("s,sld->dl", u_val, grad.data, optimize=True)
                    return VecOpInfo(data, role=grad.role, **self.update_meta(meta))
                else: # scalar multiply with grad result into gradient
                    data = u_val[0] * grad.data
                    return GradOpInfo(data, role=grad.role, **self.update_meta(meta))

        elif self.role == "trial" and grad.role == "function":
            # Case:  Trial · Grad(Function)      u_trial · ∇u_k
            # (1)  value of u_trial at this quad-point
            grad_val = _collapsed_grad(grad)  # shape (k, d)  —   ∇u_k(ξ)
            # (2)  w_i,n = Σ_d ∂_d φ_{k,n} u_d
            if self.data.shape[0] == grad_val.shape[0]:           
                data = np.einsum("sl,sd->dl", self.data, grad_val, optimize=True) # it should be (l,d) instead of (d,l)
                meta = _resolve_meta(self.meta(), meta_grad, prefer='a')
                return VecOpInfo(data, role=self.role, **self.update_meta(meta))
            else: raise NotImplementedError(self._error_msg(grad, "dot_grad"))
        elif self.role == "function" and grad.role == "function":
            # Case:  Function · Grad(Function)      u_k · ∇u_k
            # (1)  value of u_k at this quad-point
            u_val = _collapsed_function(self)  # shape (k,)  —   u_k(ξ)
            grad_val = _collapsed_grad(grad)  # shape (k, d)  —   ∇u_k(ξ)
            if u_val.shape[0] == grad_val.shape[0]:
                data = np.einsum("s,sd->d", u_val, grad_val, optimize=True)
                return data
            else: raise NotImplementedError(self._error_msg(grad, "dot_grad"))
        raise NotImplementedError(f"VecOpInfo.dot_grad not implemented for role {self.role} and GradOpInfo role {grad.role}.")
    
    def dot_vec(self, other_vec: "VecOpInfo") -> "VecOpInfo":
        """
        Computes dot(u, v) for a vector basis function u and a other vector basis.
        Returns a new VecOpInfo with shape (n,).
        """
        # print(f" a.shape = {self.data.shape}, b.shape = {vec.data.shape}")
        if not isinstance(other_vec, VecOpInfo):
            raise TypeError(f"Expected VecOpInfo, got {type(other_vec)}.")
        if other_vec.shape[0] != self.data.shape[0]:
            raise ValueError(f"Input vector of shape {other_vec.shape} cannot be dotted with VecOpInfo of shape {self.data.shape}.")
        
        # case 1 function dot test
        if  self.role == "function" and other_vec.role in {"test", "trial"}: # rhs time derivative term
            func_values_at_qp = _collapsed_function(self)  # shape (k,)
            meta = _resolve_meta(self.meta(), other_vec.meta(), prefer='b')
            if func_values_at_qp.shape[0] > 1:
                if other_vec.shape[0] == 1:
                    # Special case: single component test function
                    data = np.asarray([other_vec.data[0,:] * comp for comp in func_values_at_qp])
                    return VecOpInfo(data,
                                    role=other_vec.role, **self.update_meta(meta))
                data = np.einsum("k,kn->n", func_values_at_qp, other_vec.data, optimize=True)
                return self._expand_axis_lhs(data, other_vec.role, meta)
            elif func_values_at_qp.shape[0] == 1:
                if other_vec.shape[0] == 1:
                    # normal dot product of two scalar functions
                    data = np.einsum("k,kn->n", func_values_at_qp, other_vec.data, optimize=True)
                    return self._expand_axis_lhs(data, other_vec.role, meta)
                else: # scalar multiply with test result into vector
                    data = func_values_at_qp[0] * other_vec.data
                    return VecOpInfo(data, role=other_vec.role, **self.update_meta(meta))
            
        # case 2 trial dot test and test dot trial
        if self.role == "trial" and other_vec.role == "test":
            # lhs mass matrix term
            # If self is trial and vec is test, we return a VecOpInfo with shape (m, n)
            meta = _resolve_meta(self.meta(), other_vec.meta())
            return np.einsum("km,kn->mn", other_vec.data , self.data, optimize=True)
        elif self.role == "test" and other_vec.role == "trial":
            meta = _resolve_meta(self.meta(), other_vec.meta())
            return np.einsum("km,kn->mn", self.data, other_vec.data, optimize=True)
        
        # case 3 trial and function
        if self.role in {"trial", "test"} and other_vec.role == "function":
            meta = _resolve_meta(self.meta(), other_vec.meta(), prefer='a')
            v_values = _collapsed_function(other_vec)  # shape (k,)
            if self.shape[0] == 1 and v_values.shape[0] > 1:
                # Special case: single component trial function
                data = np.asarray([self.data[0,:] * comp for comp in v_values])
                return VecOpInfo(data, role=self.role, **self.update_meta(meta))
            elif self.shape[0] == v_values.shape[0]:
                data = np.einsum("kn,k->n", self.data, v_values, optimize=True)
                return VecOpInfo(data[np.newaxis,:], role=self.role, **self.update_meta(meta))
            elif self.shape[0] > 1 and v_values.shape[0] == 1:
                # Special case: single component function
                data = v_values[0] * self.data
                return VecOpInfo(data, role=self.role, **self.update_meta(meta))
            else:
                raise NotImplementedError(self._error_msg(other_vec, "dot_vec"))
        # case 4 function and function
        if self.role == "function" and other_vec.role == "function":
            u_values = _collapsed_function(self)  # shape (k,)
            v_values = _collapsed_function(other_vec)  # shape (k,)
            return np.dot(u_values, v_values)  # scalar result for rhs
        raise NotImplementedError(f"VecOpInfo.dot_vec not implemented for roles {self.role} and {other_vec.role}.")
    # ========================================================================
    # Shape, len, and ndim methods
    def __len__(self) -> int:
        """Returns the size of the first dimension (number of components)."""
        return self.data.shape[0] if self.data.ndim > 0 else 1
    def __mul__(self, other: Union[float, np.ndarray]) -> "VecOpInfo":
        """Element-wise multiplication with a scalar or vector."""
        if isinstance(other, (float, int)):
            return self._with(self.data * other, role=self.role)
        elif isinstance(other, np.ndarray): # np.array
            if other.ndim == 1 and other.size == self.data.shape[0]:
                return self._with(self.data * other[:, np.newaxis], role=self.role)
            elif other.ndim == 1 and self.data.shape[0] == 1:
                # New Case: Scalar multiplication with a vector
                if self.role == "function":
                    vals = _collapsed_function(self)  # shape (k,)
                    return vals * other  # returns a 1D array
                return self._with(np.asarray([self.data[0,:] * comp for comp in other]), role=self.role)
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
                u_vals = _collapsed_function(self)  # shape (k,)
                v_vals = _collapsed_function(other)  # shape (k,)
                return np.dot(u_vals, v_vals)  # scalar result for rhs
            elif self.role == "trial" and other.role == "function":
                # Case: Trial * Function , dot product case
                meta = _resolve_meta(self.meta(), other.meta(), prefer='a')
                v_vals = _collapsed_function(other)  # shape (k,)
                data = np.einsum("kn,k->n", self.data, v_vals, optimize=True)
                return VecOpInfo(data[np.newaxis,:], role=self.role, **self.update_meta(meta))
            elif self.role == "function" and other.role == "trial":
                meta = _resolve_meta(self.meta(), other.meta(), prefer='b')
                # Case: Function * Trial , dot product case
                u_vals = _collapsed_function(self)  # shape (k,)
                data = np.einsum("k,kn->n", u_vals, other.data, optimize=True)
                return VecOpInfo(data[np.newaxis,:], role=other.role, **self.update_meta(meta))
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
        meta = _resolve_meta(self.meta(), other.meta(), strict=True)
        return VecOpInfo(self.data + other.data, role=self.role, **self.update_meta(meta))
    def __sub__(self, other: "VecOpInfo") -> "VecOpInfo":
        """Element-wise subtraction with another VecOpInfo."""
        if not isinstance(other, VecOpInfo):
            raise TypeError(f"Cannot subtract {type(other)} from VecOpInfo.")
        if self.data.shape != other.data.shape:
            raise ValueError("VecOpInfo shapes mismatch in subtraction.")
        meta = _resolve_meta(self.meta(), other.meta(), strict=True)
        return VecOpInfo(self.data - other.data, role=self.role, **self.update_meta(meta))
    def __neg__(self) -> "VecOpInfo":
        """Element-wise negation of the VecOpInfo."""
        return self._with(-self.data, role=self.role)
    def info(self):
        """Return the type of the data array."""
        return (f"VecOpInfo({self.data.dtype}, shape={self.data.shape}, role='{self.role}')"
                f" (parent: {self.parent_name}, side: {self.side})"
                f" (fields: {self.field_names}, sides: {self.field_sides})"
                f" is_rhs: {self.is_rhs}"
               )

    def __repr__(self) -> str:
        """String representation of the VecOpInfo."""
        return (f"VecOpInfo(shape={self.data.shape}, role='{self.role}')"
                f" (parent: {self.parent_name}, side: {self.side})"
                f" (fields: {self.field_names}, sides: {self.field_sides})"
                f" is_rhs: {self.is_rhs}"
               )


@dataclass(slots=True, frozen=True)
class GradOpInfo(BaseOpInfo):
    """Container for gradients ∇φ. Standard shape: (k, n, d).

    k: number of components of the *operand* (1 for scalar, e.g. pressure; 2 for velocity)
    n: number of local dofs on the active side / union mapping
    d: spatial dimension
    """
    coeffs: np.ndarray = field(default=None)  # (k, n) when role == "function" and data.ndim == 3

    def _with(self, data: np.ndarray, role: str | None = None, coeffs: np.ndarray | None = None) -> "GradOpInfo":
        return GradOpInfo(
            data, role=(role or self.role), coeffs=coeffs or self.coeffs,
            field_names=self.field_names, parent_name=self.parent_name,
            side=self.side, field_sides=self.field_sides, is_rhs=self.is_rhs
        )

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
                grad_vals = _collapsed_grad(self)  # (k, d)
                return self._with(grad_vals.T, role=self.role, coeffs=None)
            else: # trial and test 
                return self._with(self.data.transpose(2, 1, 0), role=self.role, coeffs=self.coeffs)
        if self.data.ndim == 2:        # (k, d) or (n, d)
            return self._with(self.data.T, role=self.role, coeffs=self.coeffs)
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
        
        # Case 1: Function · Grad(Trial)  · Grad(Function)
        if self.role == "function" and other.role in {"trial", "test"}:
            # Case:  Function · Grad(Trial)      ∇u_k · ∇u_trial
            # (k, d)   =  Σ_i  u_{k,i}  ∂_d φ_i   – true ∇u_k at this quad-point
            grad_val = _collapsed_grad(self)  # shape (k, d)  —   ∇u_k(ξ)
            if grad_val.shape[-1] == other.shape[0]:
                # (2)  matrix multiplication
                data = np.einsum("ks,snd->knd", grad_val, other.data, optimize=True)
                meta = _resolve_meta(self.meta(), other.meta(), prefer='b')
                return GradOpInfo(data, role=other.role, **self.update_meta(meta))
            else: raise NotImplementedError(self._error_msg(other, "dot between gradients"))
        elif self.role in {"trial", "test"} and other.role == "function":
            # Case:  Grad(Trial) · Grad(Function)      ∇u_trial · ∇u_k
            # (1)  value of u_trial at this quad-point
            grad_val = _collapsed_grad(other)  # shape (k, d)  —   ∇u_k(ξ)
            if self.shape[-1] == grad_val.shape[0]:
                data = np.einsum("kns,sd->knd", self.data, grad_val, optimize=True)
                meta = _resolve_meta(self.meta(), other.meta(), prefer='a')
                return GradOpInfo(data, role=self.role, **self.update_meta(meta))
            else: raise NotImplementedError(self._error_msg(other, "dot between gradients"))
        elif self.role == "function" and other.role == "function":
            # Case:  Grad(Function) · Grad(Function)      ∇u_k · ∇u_k
            # (1)  value of ∇u_k at this quad-point
            grad_val = _collapsed_grad(self)  # shape (k, d)  —   ∇u_k(ξ)
            other_grad_val = _collapsed_grad(other)  # shape (k, d)  —   ∇u_k(ξ)
            if grad_val.shape[0] == other_grad_val.shape[0]:
                # Matrix multiplication
                data = np.einsum("kd,dn->kn", grad_val, other_grad_val, optimize=True)
                meta = _resolve_meta(self.meta(), other.meta())
                return GradOpInfo(data, role=self.role, **self.update_meta(meta))
            else: raise NotImplementedError(self._error_msg(other, "dot between gradients"))
        else:
            raise NotImplementedError(f"GradOpInfo.dot not implemented for roles {self.role} and {other.role}."
                                      f" Shapes: {self.data.shape} and {other.data.shape}.")

    def left_dot(self, left_vec:np.ndarray) -> VecOpInfo:
        """
        Computes the left dot product with a constant vector or VecOpInfo over the SPATIAL dimension.
        This operation reduces the dimension and returns a VecOpInfo object.
        dot(c, ∇v) -> c ⋅ ∇v
        """
        if isinstance(left_vec, (VecOpInfo)):
            # Case:  Const · Grad(Function)      c · ∇u_k
            if self.role == "function":
                # (1)  value of ∇u_k at this quad-point
                grad_val = _collapsed_grad(self)  # shape (k, d)  —   ∇u_k(ξ)
                # (2)  dot product
                if left_vec.role in {"trial", "test"}:
                    if grad_val.shape[0] != left_vec.shape[0]:
                        raise NotImplementedError(self._error_msg(left_vec, "left_dot with vector"))
                    meta = _resolve_meta(self.meta(), left_vec.meta(), prefer='b')
                    data = np.einsum("kn,kd->dn", left_vec.data, grad_val, optimize=True)
                    return VecOpInfo(data, role=left_vec.role, **left_vec.update_meta(meta))
                elif left_vec.role == "function":
                    u_vals = _collapsed_function(left_vec)  # shape (k,)
                    return np.dot(u_vals, grad_val)  # (d,) result
            elif self.role in {"trial", "test"}:
                if left_vec.role == "function":
                    u_vals = _collapsed_function(left_vec)  # shape (k,)
                    if u_vals.shape[0] >1:
                        if self.data.shape[0] == 1:
                            # Special case: single component gradient
                            data = np.einsum("d,kld->kl", u_vals, self.data, optimize=True)
                        elif self.data.shape[0] == u_vals.shape[0]:
                            data = np.einsum("s,snd->dn", u_vals, self.data, optimize=True)
                        else: raise NotImplementedError(self._error_msg(left_vec, "left_dot with vector"))
                        meta = _resolve_meta(self.meta(), left_vec.meta(), prefer='a')
                        return VecOpInfo(data, role=self.role, **left_vec.update_meta(meta))
                    else: raise NotImplementedError(self._error_msg(left_vec, "left_dot with vector"))

        if isinstance(left_vec, np.ndarray) and left_vec.ndim == 1:
            if self.role == "function":
                if left_vec.shape[0] >1:
                    # Case:  Const · Grad(Function)      c · ∇u_k
                    grad_val = _collapsed_grad(self)  # shape (k, d)  —   ∇u_k(ξ)
                    if grad_val.shape[0] == 1:
                        # Special case: single component gradient
                        data = np.einsum("d,kd->k", left_vec, grad_val, optimize=True)
                        return data # returns a 1D array
                    else:
                        data = np.einsum("k,kd->d", left_vec, grad_val, optimize=True)
                    return data
                else:
                    raise ValueError(f"Cannot left_dot with vector of shape {left_vec.shape}.")
            elif self.role in {"trial", "test"}:
                if left_vec.shape[0] >1:
                    if self.data.shape[0] == 1:
                        # Special case: single component gradient
                        data = np.einsum("d,kld->kl", left_vec, self.data, optimize=True)
                    else:
                        data = np.einsum("s,snd->dn", left_vec, self.data, optimize=True)
                    meta = _resolve_meta(self.meta(), {}, prefer='a')
                    return VecOpInfo(data, role=self.role, **self.update_meta(meta))
                else:
                    raise ValueError(f"Cannot left_dot with vector of shape {left_vec.shape}.")

        raise ValueError(f"Cannot left_dot with vector of shape {left_vec.shape}.")

    def dot_vec(self, other_vec: np.ndarray) -> VecOpInfo:
        """
        Computes the dot product with a constant vector or VecOpInfo over the SPATIAL dimension.
        This operation reduces the dimension and returns a VecOpInfo object.
        dot(∇v, c) -> ∇v ⋅ c
        """
        
        if isinstance(other_vec, (VecOpInfo)): # this part is until trial grad(u) dot u_k  ((∇u) · u_k)
            # role = self.role
            # if vec.role == "trial":
            #     role = vec.role
            # if vec.data.shape[0] != self.data.shape[-1]:
            #     raise ValueError(f"Cannot dot GradOpInfo {self.shape} with VecOpInfo of shape {vec.data.shape}.")
            # result_data = np.einsum("ijk,kl->ij", self.data, vec.data, optimize=True)
            # return VecOpInfo(result_data, role=role)
            if self.role == "function" and other_vec.role == "trial": # introducing a new branch
                # Case:  Grad(Function) · Vec(Trial)      (∇u_k) · u
                grad_val = _collapsed_grad(self)  # shape (k, d)  —   ∇u_k(ξ)
                if grad_val.shape[-1] == other_vec.shape[0]:
                    data = grad_val @ other_vec.data
                    meta = _resolve_meta(self.meta(), other_vec.meta(), prefer='b')
                    return VecOpInfo(data, role=other_vec.role, **other_vec.update_meta(meta))
                else: raise NotImplementedError(self._error_msg(other_vec, "dot_vec"))
            
            if self.role == "trial" and other_vec.role == "function": # introducing a new branch
                # Case:  Grad(Trial) · Vec(Function)      (∇u_trial) · u_k
                v_val = _collapsed_function(other_vec)  # shape (k,)  —   u_k(ξ)
                if self.data.shape[-1] != v_val.shape[0]:
                    raise NotImplementedError(self._error_msg(other_vec, "dot_vec"))
                data = np.einsum("kld,d->kl", self.data, v_val, optimize=True)
                meta = _resolve_meta(self.meta(), other_vec.meta(), prefer='a')
                return VecOpInfo(data, role=self.role, **other_vec.update_meta(meta))
            if self.role == "function" and other_vec.role == "function":
                # Case:  Grad(Function) · Vec(Function)      (∇u_k) · u_k
                # (1)  value of ∇u_k at this quad-point
                grad_val = _collapsed_grad(self)  # shape (k, d)  —   ∇u_k(ξ)
                v_val = _collapsed_function(other_vec)  # shape (k,)  —   u_k(ξ)
                if grad_val.shape[-1] != v_val.shape[0]:
                    raise NotImplementedError(self._error_msg(other_vec, "dot_vec"))
                data = np.einsum("kd,d->k", grad_val, v_val, optimize=True)
                return data # rhs, so return a 1D array

            
        
        if isinstance(other_vec, np.ndarray) and other_vec.ndim == 1: # dot product with a constant vector
            # print(f"GradOpInfo.dot_vec: vec={vec}, data.shape={self.data.shape}, role={self.role}"
                #   f", vec.role={getattr(vec, 'role', None)}, vec.shape={getattr(vec, 'shape', None)}")
            if self.role == "function":
                # Case:  Grad(Function) · Const      (∇u_k) · c
                # (1)  value of ∇u_k at this quad-point
                grad_val = _collapsed_grad(self)  # shape (k, d)  —   ∇u_k(ξ)
                # (2)  w_i,n = Σ_d (∇u_k)_d c_d φ_{k,n}
                if grad_val.shape[-1] != other_vec.shape[0]:
                    raise NotImplementedError(self._error_msg(other_vec, "dot_vec"))
                data = np.einsum("kd,d->k", grad_val, other_vec, optimize=True)                
                return data # returns a 1D array
            if self.data.shape[-1] != other_vec.shape[0]:
                raise ValueError(f"Cannot dot GradOpInfo {self.shape} with vector of shape {other_vec.shape}.")
            result_data = np.einsum("knd,d->kn", self.data, other_vec, optimize=True)
            meta = _resolve_meta(self.meta(), {}, prefer='a')
            return VecOpInfo(result_data, role=self.role, **self.update_meta(meta))
        raise NotImplementedError(f"dot_vec of GradOpInfo not implemented for role {self.role}, GradOpInfo  and type {type(other_vec)} with role: {other_vec.role}.")

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

        return self._with(new_data, role=self.role)

    def __rmul__(self, other: Union[float, int, np.ndarray]) -> "GradOpInfo":
        return self.__mul__(other)
        
    def __neg__(self) -> "GradOpInfo":
        return self._with(-self.data, role=self.role)

    def __add__(self, other: "GradOpInfo") -> "GradOpInfo":
        if not isinstance(other, GradOpInfo) and self.role != other.role:
            raise ValueError("Operands must be GradOpInfo of the same shape for addition."
                             f" Shapes: {self.data.shape} and {other.data.shape}."
                             f" Roles: {self.role} and {other.role}.")
        if self.role in ["test", "trial"] and other.role in ["test", "trial"]:
            # Case: both are test or trial gradients
            if self.data.shape != other.data.shape:
                raise ValueError(f"GradOpInfo shapes mismatch in addition: {self.data.shape} vs {other.data.shape}.")
            return self._with(self.data + other.data, role=self.role)
        elif self.role == "function" and other.role == "function":
            a = self._eval_function_to_2d()  
            b = other._eval_function_to_2d()
            if a.shape != b.shape:
                raise ValueError(f"Function gradient shapes mismatch in addition: {a.shape} vs {b.shape}.")
            return self._with(a + b, role=self.role) # collapsed gradients (2,2)
        else:
            raise NotImplementedError(f"GradOpInfo addition not implemented for roles {self.role} and {other.role}."
                                      f" Shapes: {self.data.shape} and {other.data.shape}.")

    def __sub__(self, other: "GradOpInfo") -> "GradOpInfo":
        return self.__add__(-other)

    # --- Helper properties ---
    def __repr__(self): return (f"GradOpInfo(shape={self.data.shape}, role='{self.role}')"
           f" (parent: {self.parent_name}, side: {self.side})"
           f" (fields: {self.field_names}, sides: {self.field_sides})"
           f" is_rhs: {self.is_rhs}"
           )

    def info(self):
        """Return the type of the data array."""
        return (f"GradOpInfo({self.data.dtype}, shape={self.data.shape}, role='{self.role}')"
                f" (parent: {self.parent_name}, side: {self.side})"
                f" (fields: {self.field_names}, sides: {self.field_sides})"
                f" is_rhs: {self.is_rhs}"
               )



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


#-----------------------------------------------------------------------
# Field aware Helpers
#-----------------------------------------------------------------------
class HelpersFieldAware:
    @staticmethod
    def infer_side_from_field_name(field: str) -> Optional[str]:
        """Return 'pos' or 'neg' if the field name encodes a side; else None."""
        if "_pos_" in field or field.endswith("_pos") or field.startswith("pos_"):
            return "pos"
        if "_neg_" in field or field.endswith("_neg") or field.startswith("neg_"):
            return "neg"
        return None
    @staticmethod
    def elemental_field_dofs(dh: "DofHandler", eid: int, field: str) -> np.ndarray:
        """
        Return the global DOFs (np.ndarray[int]) for a given element and field.
        Relies on DofHandler.element_maps[field][eid].
        """
        try:
            return np.asarray(dh.element_maps[field][eid], dtype=int)
        except Exception as ex:
            raise KeyError(f"Missing elemental DOFs for field '{field}' on element {eid}") from ex

    @staticmethod
    def build_field_union_maps(
            dh: "DofHandler",
            fields: Sequence[str],
            pos_eid: int,
            neg_eid: int,
            global_dofs: np.ndarray,
        ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Build per-field padding maps from field-local DOF ids to union columns.
        Robust to non-sorted `global_dofs` by using an id→index map.
        """
        pos_map_by_field: Dict[str, np.ndarray] = {}
        neg_map_by_field: Dict[str, np.ndarray] = {}
        union = np.asarray(global_dofs, dtype=int)
        col_of = {int(d): j for j, d in enumerate(union)}
        for fld in fields:
            # Positive side
            try:
                local_gdofs = HelpersFieldAware.elemental_field_dofs(dh, int(pos_eid), fld)
                pos_map_by_field[fld] = np.asarray([col_of[int(d)] for d in local_gdofs], dtype=int)
            except Exception:
                pass
            # Negative side
            try:
                local_gdofs = HelpersFieldAware.elemental_field_dofs(dh, int(neg_eid), fld)
                neg_map_by_field[fld] = np.asarray([col_of[int(d)] for d in local_gdofs], dtype=int)
            except Exception:
                pass
        return pos_map_by_field, neg_map_by_field

    @staticmethod
    def get_field_map(ctx: dict, side: str, field: str) -> Optional[np.ndarray]:
        """
        Fetch the per-field padding map for the given side ('+' or '-') and field name.
        Returns None if the collection isn't present or isn't a dict.
        """
        key = "pos_map_by_field" if side == '+' else "neg_map_by_field"
        by_field = ctx.get(key)
        if not isinstance(by_field, dict):
            return None
        return by_field.get(field)
    @staticmethod
    def build_side_masks_by_field(
        dh: "DofHandler",
        fields: Sequence[str],
        eid: int,
        level_set,
        tol: float = 0.0,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Per-element masks: for each field, mark which local DOFs lie on φ>=-tol ('+') vs φ<-tol ('−').
        """
        coords = dh.get_all_dof_coords()
        pos_masks: Dict[str, np.ndarray] = {}
        neg_masks: Dict[str, np.ndarray] = {}
        for fld in fields:
            try:
                gidx = np.asarray(dh.element_maps[fld][eid], dtype=int)
            except Exception:
                continue
            xy   = coords[gidx]
            from pycutfem.ufl.helpers_geom import phi_eval
            phi  = np.asarray([phi_eval(level_set, p) for p in xy], dtype=float)
            pos_masks[fld] = (phi >= -float(tol)).astype(float)
            neg_masks[fld] = (phi <  -float(tol)).astype(float)
        return pos_masks, neg_masks
    @staticmethod
    def _side_from_node(self:"FormCompiler", n):
        s = getattr(n, "side", "")
        if s in ("+","-"):           # 1) explicit side set by user
            return s
        fs = getattr(n, "field_sides", None)  # 2) all components declare the same side?
        if isinstance(fs, list) and fs:
            tags = {t for t in fs if t is not None}
            if tags == {"pos"}: return "+"
            if tags == {"neg"}: return "-"
        return None                   # 3) unknown
    @staticmethod
    def _filter_fields_for_side(self:"FormCompiler", n, fields):
        side = HelpersFieldAware._side_from_node(n)
        if side in ("+","-"):
            keep = "pos" if side == "+" else "neg"
            return [f for f in fields
                    if (HelpersFieldAware.infer_side_from_field_name(f) in (None, keep))]
        # fallback: infer from names (only if consistent)
        tags = {HelpersFieldAware.infer_side_from_field_name(f) for f in fields}
        tags.discard(None)
        if len(tags) == 1:
            keep = next(iter(tags))
            return [f for f in fields
                    if (HelpersFieldAware.infer_side_from_field_name(f) in (None, keep))]
        return fields