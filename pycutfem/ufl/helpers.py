import re
import numpy as np
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Union, Tuple, Set, Sequence, Optional, Dict, Any
from pycutfem.ufl.expressions import Expression, Derivative, Constant, Identity
from pycutfem.ufl.expressions import (
    VectorFunction, TrialFunction, VectorTrialFunction,
    TestFunction, VectorTestFunction, Restriction,
    Grad as UFLGrad,
    DivOperation as UFLDiv,
)
from pycutfem.ufl.forms import Form, Equation
from pycutfem.core.dofhandler import DofHandler
from pycutfem.fem.mixedelement import MixedElement
from pycutfem.ufl.expressions import Hessian as UFLHessian, Laplacian as UFLLaplacian
from pycutfem.ufl.helpers_geom import phi_eval
from pycutfem.core.sideconvention import SIDE



import logging
# Setup logging
logger = logging.getLogger(__name__)

def build_l2_projector(B: np.ndarray, w: np.ndarray, J: np.ndarray) -> np.ndarray:
    """
    Construct the local L² projector P = C M^{-1} that maps the full trace
    basis represented by columns of B onto the subspace spanned by the DOFs
    indexed by J.

    Parameters
    ----------
    B : ndarray (n_q, n)
        Basis values on the interface for all field-local DOFs.
    w : ndarray (n_q,)
        Quadrature weights on the interface segment (already scaled by
        geometric factors as used during assembly).
    J : ndarray (n_s,)
        Indices (within the field-local ordering) that belong to the side
        subspace.
    """
    B = np.asarray(B, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1, 1)
    J = np.asarray(J, dtype=int)

    if B.size == 0 or J.size == 0:
        return np.zeros((B.shape[1], 0), dtype=float)

    Bw = B * w
    Bs = B[:, J]
    Bsw = Bs * w

    Ms = Bs.T @ Bsw
    Cs = B.T @ Bsw

    if Ms.size == 0:
        return np.zeros((B.shape[1], 0), dtype=float)

    tau = 1e-12 * float(np.trace(Ms)) / max(1, Ms.shape[0])
    Ms_reg = Ms + tau * np.eye(Ms.shape[0], dtype=Ms.dtype)
    proj = Cs @ np.linalg.solve(Ms_reg, np.eye(Ms.shape[0], dtype=Ms.dtype))
    print("build_l2_projector", B.shape, J.size)
    return proj
def _meta_has(val, kind, member_names = ("field_names", "parent_name", "side", "field_sides", "is_rhs")):
    for name in member_names:
        if name == kind: return bool(val)
    return False
def _resolve_meta(a, b, prefer=None, strict=False):
    """
    Merge metadata dicts {'field_names','parent_name','side','field_sides'}.
    prefer ∈ {None,'a','b'}; strict -> raise on field_names/parent_name conflict.
    """
    if a is None:
        a = {}
    if b is None:
        b = {}
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
# ---------- collapsed helpers (accept OpInfo or ndarray) ----------
def _collapsed_function(a: Union["VecOpInfo", np.ndarray]) -> np.ndarray:
    """(k,n)->(k,), (k,)->(k,), scalar->(1,) — keeps vector-of-components."""
    if isinstance(a, GradOpInfo) or isinstance(a, HessOpInfo):
        raise ValueError("_collapsed_function: expected VecOpInfo or ndarray, got GradOpInfo/HessOpInfo.")
    A = a.data if isinstance(a, VecOpInfo) else a
    A = np.asarray(A)
    if A.ndim == 2 and A.shape[0] != A.shape[1]:      # (k,n)
        return A.sum(axis=1)
    elif A.ndim == 1:      # (k,)
        return A
    elif A.ndim == 0:      # scalar -> length-1 vector
        return A.reshape(1,)
    else:
        role = getattr(a, 'role', None)
        type_a = a.__class__.__name__ if isinstance(a, BaseOpInfo) else type(a)
        raise ValueError(f"_collapsed_function: unexpected shape {A.shape} for shape with role {role}."
                         f" Type: {type_a}.")

def _collapsed_grad(g: Union["GradOpInfo", np.ndarray]) -> np.ndarray:
    """(k,n,d)+coeffs -> (k,d); pass-through (k,d)."""
    if isinstance(g, VecOpInfo) or isinstance(g, HessOpInfo):
        raise ValueError("_collapsed_grad: expected GradOpInfo or ndarray, got VecOpInfo/HessOpInfo.")
    if isinstance(g, GradOpInfo):
        G = np.asarray(g.data)
        if G.ndim == 3:
            if g.coeffs is None:
                raise ValueError("_collapsed_grad: coeffs required for (k,n,d).")
            # (k,n,d) x (k,n) -> (k,d)
            return np.einsum("knd,kn->kd", G, g.coeffs, optimize=True)
        if G.ndim == 2:
            return G
        if G.ndim == 1:
            # Scalar gradients may be stored as a plain spatial vector (d,).
            return G[np.newaxis, :]
        raise ValueError(f"_collapsed_grad: unexpected grad data shape {G.shape}")
    G = np.asarray(g)
    if G.ndim == 2:
        return G
    if G.ndim == 1:
        return G[np.newaxis, :]
    raise ValueError(f"_collapsed_grad: unexpected ndarray shape {G.shape}")


def _scalar_grad_basis_matrix(g: Union["GradOpInfo", np.ndarray]) -> np.ndarray:
    """
    Return scalar basis gradients in canonical carried form ``(d, n)``.

    Accepts both the legacy ``(1, n, d)`` storage and the refactored ``(d, n)``
    storage so the algebra layer can migrate without breaking older paths.
    """
    G = np.asarray(g.data if isinstance(g, GradOpInfo) else g)
    if G.ndim == 2:
        return G if G.shape[0] <= G.shape[1] else G.T
    if G.ndim == 3 and G.shape[0] == 1:
        return np.ascontiguousarray(G[0].T)
    raise ValueError(f"_scalar_grad_basis_matrix: unexpected gradient basis shape {G.shape}")
def _collapsed_hess(h: Union["HessOpInfo", np.ndarray]) -> np.ndarray:
    """(k,n,d,d)+coeffs -> (k,d,d); pass-through (k,d,d)."""
    if isinstance(h, VecOpInfo) or isinstance(h, GradOpInfo):
        raise ValueError("_collapsed_hess: expected HessOpInfo or ndarray, got VecOpInfo/GradOpInfo.")
    if isinstance(h, HessOpInfo):
        H = np.asarray(h.data)
        if H.ndim == 4:
            if h.coeffs is None:
                raise ValueError("_collapsed_hess: coeffs required for (k,n,d,d).")
            # (k,n,d1,d2) x (k,n) -> (k,d1,d2)
            return np.einsum("knij,kn->kij", H, h.coeffs, optimize=True)
        if H.ndim == 3:
            return H
        raise ValueError(f"_collapsed_hess: unexpected hess data shape {H.shape}")
    H = np.asarray(h)
    if H.ndim == 3: return H
    raise ValueError(f"_collapsed_hess: unexpected ndarray shape {H.shape}")

def _is_1d_vector(vec) -> bool:
    """Check if the given vector is 1D."""
    if isinstance(vec,np.ndarray):
        if vec.ndim == 1 and vec.shape[0] > 1: return True
        elif vec.ndim == 2 and vec.shape[0] > 1 and vec.shape[1] == 1: return True
        elif vec.ndim == 3 and vec.shape[0] > 1 and vec.shape[1] == 1 and vec.shape[2] == 1: return True
        elif vec.ndim > 3: raise NotImplementedError(f"{vec.ndim}D arrays not supported for 1D vector check.")
        else: return False
    if isinstance(vec,VecOpInfo): return _is_1d_vector(vec.data)
    if isinstance(vec,GradOpInfo): return _is_1d_vector(vec.data)
    raise NotImplementedError(f"Unsupported vector type: {type(vec)}")
def _is_scalar(a) -> bool:
    """Check if the given value is a scalar (treat length-1 arrays as scalar)."""
    if isinstance(a,np.ndarray):
        if a.ndim == 0: return True
        elif a.ndim == 1 and a.shape[0] == 1: return True
        elif a.ndim == 2 and a.shape[0] == 1 and a.shape[1] == 1: return True
        elif a.ndim == 3 and a.shape[0] == 1 and a.shape[1] == 1 and a.shape[2] == 1: return True
        elif a.ndim > 3: raise NotImplementedError(f"{a.ndim}D arrays not supported for scalar check.")
        else: return False
    if isinstance(a, VecOpInfo): return _is_scalar(a.data)
    if isinstance(a, GradOpInfo): return _is_scalar(a.data)
    if np.ndim(a) == 0: return True
    raise NotImplementedError(f"Unsupported type: {type(a)}")
def _is_scalar_field(a) -> bool:
    """Check if the given vector is a scalar field.
       Fields can be trial and test for function if we have only one component."""
    if isinstance(a, VecOpInfo):
        if a.role in {"trial", "test"}:
            return a.data.shape[0] == 1
        elif a.role == "function":
            collapsed_fun = _collapsed_function(a.data)
            return collapsed_fun.shape[0] == 1
        # elif a.role in {"scalar"}:
        #     return True
        # elif a.role in {"vector"}:
        #     return False
        # elif a.role in {"scalar", "vector"}:
        #     return False
        else:
            raise NotImplementedError(f"Unsupported VecOpInfo role: {a.role}")
    if isinstance(a, GradOpInfo):
        # print(f"GradOpInfo detected: a.role: {a.role}, a.shape: {a.shape}")
        if a.role in {"trial", "test"}:
            return a.data.ndim == 2 or a.data.shape[0] == 1
        elif a.role == "function":
            grad_collapsed = _collapsed_grad(a.data)
            return grad_collapsed.shape[0] == 1
        # elif a.role in {"scalar", "vector"}:
        #     return False
        else:
            raise NotImplementedError(f"Unsupported GradOpInfo role: {a.role}")
    if isinstance(a, HessOpInfo):
        if a.role in {'trial', 'test', 'function'}:
            return a.data.shape[0] == 1
        else:
            raise NotImplementedError(f"Unsupported HessOpInfo role: {a.role}")
    # if isinstance(a,np.ndarray):
    #     return a.ndim == 0
    raise NotImplementedError(f"Unsupported type: {type(a)}")

# ========================================================================
#  Tensor Containers for Symbolic Basis Functions
# ========================================================================
def lhs_num(value: Any) -> np.ndarray:
    """
    Return a numeric view suitable for left-hand side assembly.

    For BaseOpInfo instances with is_rhs=False we strip the leading
    component axis when it is a singleton (e.g. shapes (1,n), (1,n,m),
    (1,n,d), …) so the result can accumulate into plain numpy arrays
    without broadcasting issues. All other inputs are coerced with
    np.asarray.
    """
    if isinstance(value, BaseOpInfo):
        view = np.asarray(value.data)
        while isinstance(view, np.ndarray) and view.ndim > 0 and view.shape[0] == 1:
            view = view[0]
        return view
    return np.asarray(value)


def _apply_storage_transform_opinfo(item: Any, transform: Any):
    from pycutfem.ufl.tensor_algebra import OperandTransform

    if transform == OperandTransform.NONE or not isinstance(item, BaseOpInfo):
        return item

    data = np.asarray(item.data)
    if transform == OperandTransform.TRANSPOSE_2D:
        if data.ndim != 2:
            raise ValueError(f"transpose_2d storage transform expects rank-2 data, got {data.shape!r}.")
        new_data = np.ascontiguousarray(data.T)
    elif transform == OperandTransform.SCALAR_GRAD_TO_VECTOR:
        if data.ndim != 3 or int(data.shape[0]) != 1:
            raise ValueError(
                f"scalar_grad_to_vector storage transform expects shape (1,n,d), got {data.shape!r}."
            )
        new_data = np.ascontiguousarray(data[0].T)
    else:
        raise ValueError(f"Unsupported storage transform {transform!r}.")

    if isinstance(item, GradOpInfo):
        return item._with(new_data, role=item.role, coeffs=item.coeffs)
    if isinstance(item, HessOpInfo):
        return item._with(new_data, role=item.role, coeffs=item.coeffs)
    if isinstance(item, VecOpInfo):
        return item._with(new_data, role=item.role)
    return item


def _tensor_rank_of_opinfo(item: Any, *, spatial_dim: int = 2) -> int | None:
    try:
        from pycutfem.ufl.tensor_algebra import TensorRuleEngine

        return int(TensorRuleEngine.infer_signature(item, spatial_dim=spatial_dim).tensor_rank)
    except Exception:
        return None

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
        shape = getattr(self.data, 'shape', ())
        return shape
    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the data array."""
        ndim = getattr(self.data, 'ndim', None)
        if ndim is not None:
            return ndim
        else:
            return 0
    # ---------- NumPy interop ----------
    # Make NumPy prefer our ufunc handler over eager coercion to ndarray
    __array_priority__ = 10_000

    def __array__(self, dtype=None):
        """Expose numeric view (lhs-aware) when NumPy coerces this object."""
        arr = lhs_num(self)
        return np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Preserve metadata for mul/div; allow numeric add/sub with ndarray
        accumulators during assembly without leaking types elsewhere.
        """
        if method != "__call__":
            return NotImplemented

        def _num(x):
            # numeric view for accumulation; squeeze (1,n)->(n,) for vectors
            if isinstance(x, BaseOpInfo):
                return lhs_num(x)
            return x
        
        # Allow numeric accumulation with ndarray on either side
        if ufunc in (np.add, np.subtract):
            if any(not isinstance(i, BaseOpInfo) for i in inputs):
                args = tuple(_num(i) for i in inputs)
                return ufunc(*args, **kwargs)
            # let VecOpInfo/GradOpInfo/HessOpInfo.__add__/__sub__ handle typed sums
            return NotImplemented

        # Preserve types & metadata for multiply
        if ufunc is np.multiply:
            a, *rest = inputs
            b = rest[0] if rest else None
            if isinstance(a, BaseOpInfo) and isinstance(b, BaseOpInfo):
                return a.__mul__(b)
            if isinstance(a, BaseOpInfo):
                return a.__mul__(b)
            if isinstance(b, BaseOpInfo):
                return b.__rmul__(a)
            return NotImplemented

        # Preserve types & metadata for true divide
        if ufunc in (np.true_divide, np.divide):
            a, *rest = inputs
            b = rest[0] if rest else None
            if isinstance(a, BaseOpInfo) and not isinstance(b, BaseOpInfo):
                return a.__truediv__(b)
            if isinstance(b, BaseOpInfo) and not isinstance(a, BaseOpInfo):
                return b.__rtruediv__(a)
            return NotImplemented

        # Negation
        if ufunc is np.negative and len(inputs) == 1 and inputs[0] is self:
            return -self

        return NotImplemented
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


@dataclass(slots=True, frozen=True)
class VecOpInfo(BaseOpInfo):
    """Container for vector basis functions φ (phi). Shape: (k, n)."""
    # data shape: (num_components=k, num_local_dofs=n)
    def _with(self, data: np.ndarray, role: str | None = None) -> "VecOpInfo":
        return VecOpInfo(np.asarray(data), role=(role or self.role),
                        field_names=self.field_names,
                        parent_name=self.parent_name, side=self.side,
                        field_sides=self.field_sides, is_rhs=self.is_rhs)
    
    def _expand_axis_lhs(self, data: np.ndarray, role: str, meta: dict):
        if self.is_rhs: 
            return VecOpInfo(data, role=f"{role}_n", **self.update_meta(meta)) # (n,)
        else: # lhs
            return VecOpInfo(data[np.newaxis,:], role=role, **self.update_meta(meta)) # (1,n)

    def _matrix_contract_left(self, other: np.ndarray) -> "BaseOpInfo":
        mat = np.asarray(other, dtype=float)
        if mat.ndim != 2:
            raise NotImplementedError(self._error_msg(other, "matrix_contract_left"))

        meta = self.update_meta(self.meta())
        if self.role in {"function", "vector"}:
            vals = _collapsed_function(self) if self.role == "function" else np.asarray(self.data, dtype=float)
            vals = np.asarray(vals, dtype=float).reshape(-1)
            if mat.shape[1] != vals.shape[0]:
                raise NotImplementedError(self._error_msg(other, "matrix_contract_left"))
            data = mat @ vals
            role = "scalar" if np.ndim(data) == 0 or np.asarray(data).shape == (1,) else "vector"
            return VecOpInfo(np.asarray(data), role=role, **meta)

        if self.role in {"trial", "test", "trial_n", "test_n"}:
            basis = np.asarray(self.data, dtype=float)
            if basis.ndim == 2 and mat.shape[1] == basis.shape[0]:
                data = np.einsum("ij,jn->in", mat, basis, optimize=True)
                return VecOpInfo(data, role=self.role, **meta)
            raise NotImplementedError(self._error_msg(other, "matrix_contract_left"))

        if self.role == "mixed":
            mixed = np.asarray(self.data, dtype=float)
            if mixed.ndim == 3 and mat.shape[1] == mixed.shape[0]:
                data = np.einsum("ij,jmn->imn", mat, mixed, optimize=True)
                return VecOpInfo(data, role=self.role, **meta)
            raise NotImplementedError(self._error_msg(other, "matrix_contract_left"))

        raise NotImplementedError(self._error_msg(other, "matrix_contract_left"))

    def _matrix_contract_right(self, other: np.ndarray) -> "BaseOpInfo":
        mat = np.asarray(other, dtype=float)
        if mat.ndim != 2:
            raise NotImplementedError(self._error_msg(other, "matrix_contract_right"))

        meta = self.update_meta(self.meta())
        if self.role in {"function", "vector"}:
            vals = _collapsed_function(self) if self.role == "function" else np.asarray(self.data, dtype=float)
            vals = np.asarray(vals, dtype=float).reshape(-1)
            if vals.shape[0] != mat.shape[0]:
                raise NotImplementedError(self._error_msg(other, "matrix_contract_right"))
            data = vals @ mat
            role = "scalar" if np.ndim(data) == 0 or np.asarray(data).shape == (1,) else "vector"
            return VecOpInfo(np.asarray(data), role=role, **meta)

        if self.role in {"trial", "test", "trial_n", "test_n"}:
            basis = np.asarray(self.data, dtype=float)
            if basis.ndim == 2 and basis.shape[0] == mat.shape[0]:
                data = np.einsum("in,ij->jn", basis, mat, optimize=True)
                return VecOpInfo(data, role=self.role, **meta)
            raise NotImplementedError(self._error_msg(other, "matrix_contract_right"))

        if self.role == "mixed":
            mixed = np.asarray(self.data, dtype=float)
            if mixed.ndim == 3 and mixed.shape[0] == mat.shape[0]:
                data = np.einsum("imn,ij->jmn", mixed, mat, optimize=True)
                return VecOpInfo(data, role=self.role, **meta)
            raise NotImplementedError(self._error_msg(other, "matrix_contract_right"))

        raise NotImplementedError(self._error_msg(other, "matrix_contract_right"))
    

    def inner(self, other: Optional[Union["VecOpInfo", np.ndarray]]) -> np.ndarray:
        """Computes inner product (u, v), returning an (n, n) matrix."""
        role_a = self.role
        if isinstance(other, VecOpInfo):
            B = other.data
            role_b = other.role
        else:
            B = np.asarray(other)
            role_b = None

        A = self.data
        other_shape = getattr(other, 'shape', None)
        type_a = self.__class__.__name__
        type_b = other.__class__.__name__
        A = np.asarray(A)
        B = np.asarray(B)
        is_A_scalar = A.ndim == 0
        is_B_scalar = B.ndim == 0
        def is_field_1d(role, field, arr: np.ndarray) -> bool:
            """
            Return True when the operand should be treated as a 1D vector in
            scalar-multiplication branches.

            - trial/test/function: True iff it is a *scalar* field (1 component)
            - vector / None      : True iff the numeric array is 1D
            - scalar             : False (scalar is handled via `is_*_scalar`)
            """
            if role in {"trial", "test", "function"}:
                return _is_scalar_field(field)
            if role in {"vector", None}:
                return np.asarray(arr).ndim == 1
            if role == "scalar":
                return False
            raise ValueError(self._error_msg(other, "VecOpInfo.inner"))

        is_A_1d = is_field_1d(role_a, self, A)
        is_B_1d = is_field_1d(role_b, other, B)
        if A.ndim == 2 and B.ndim == 1 and A.shape[0] != B.shape[0]:
            raise ValueError("VecOpInfo component mismatch for inner product.")
        if A.ndim == 2 and B.ndim == 0 and A.shape[0] != 1:
            raise ValueError("VecOpInfo component mismatch for inner product.")
        if self.is_rhs:
            if role_a in {"trial","test"} and role_b in {"function"}:
                fun = _collapsed_function(other) # (k,)
                return np.einsum("kn,k->n", A, fun, optimize=True) # (n,)
            elif role_a in {"function"} and role_b in {"trial","test"}:
                fun = _collapsed_function(self) # (k,)
                return np.einsum("kn,k->n", B, fun, optimize=True) # (n,)
            elif role_a in {"function"} and role_b in {"function"}:
                fun_a = _collapsed_function(self) # (k,)
                fun_b = _collapsed_function(other) # (k,)
                return np.einsum("k,k->", fun_a, fun_b, optimize=True) # ()
            elif role_a in {"trial","test"} and role_b is None and B.ndim == 1:
                # Test/trial dotted with a constant vector (e.g. interface normal).
                return np.einsum("kn,k->n", A, B, optimize=True)
            elif role_a in {"trial","test"} and role_b is None and B.ndim == 0:
                # Scalar test/trial times scalar constant.
                return A[0, :] * float(B)
            elif role_a in {"vector", "trial","test"} and role_b in {"scalar"} and is_B_scalar and is_A_1d:
                return A.flatten() * float(B)
            elif role_a in {"scalar"} and role_b in {"vector", "trial","test"} and is_A_scalar and is_B_1d:
                return B.flatten() * float(A)
            elif role_a in {"vector"} and role_b in {"vector"}:
                return np.dot(A, B)
            elif role_a in {"vector"} and role_b in {"function"}:
                fun = _collapsed_function(other)
                return np.dot(A, fun)
            elif role_a in {"function"} and role_b in {"vector"}:
                fun = _collapsed_function(self)
                return np.dot(B, fun)
            elif role_a in {"vector"} and role_b in {"test"}:
                return np.dot(A, B)
            elif role_a in {"test"} and role_b in {"vector"}:
                return np.dot(B, A)
            elif role_a == None and A.ndim == 1:
                if role_b in {"function"}:
                    fun = _collapsed_function(other)
                    return np.dot(A, fun)
                elif role_b in {"vector"}:
                    return np.dot(A, B)
                elif role_b in {"trial","test"}:
                    return np.einsum("k,kn->n", A, B, optimize=True)
                elif role_b == None:
                    return float(np.einsum("..., ...->", A, B, optimize=True)) if A.ndim == B.ndim == 1 else np.tensordot(A, B, axes=([0], [0]))
                else:
                    raise ValueError(self._error_msg(other, "VecOpInfo.inner"))

        else:
            if role_a in {"trial","test"} and role_b in {"trial","test"}:
                test_var = self if role_a == "test" else other
                trial_var = other if role_a == "test" else self
                return test_var.data.T @ trial_var.data # (n,n)

        raise ValueError(f"Unsupported inner dims shape_A:{self.shape}, shape_B:{other_shape} for VecOpInfo."
                         f", Roles: A={role_a}, B={role_b}."
                         f", is_rhs: {self.is_rhs}"
                         f", Type: {type_a}/{type_b}"
                         f", is_A_1d: {is_A_1d}, is_B_1d: {is_B_1d}")
    

    def dot_const(self, const: np.ndarray):
        """Computes dot(v, c), returning an (n,) vector."""
        # case for Constant with dim = 1 and also normal vector
        logger.debug(f"VecOpInfo.dot_const: const={const}, data.shape={self.data.shape}")
        const = np.asarray(const)
        if self.role == "function":
            vals = _collapsed_function(self.data)
            if const.shape[0] == vals.shape[0]:
                data = np.dot(vals, const)  # scalar result
                role = "scalar" if data.ndim == 0 else "vector"
                return self._with(data, role=role)
            elif vals.shape[0] == 1 and _is_1d_vector(const): # scalar function dot with vector
                # make it to vector function by broadcasting
                data = vals[0] * const # (k_vec,)
                return self._with(data, role=self.role)
        elif self.role in {"trial","test"}:
            if _is_scalar_field(self) and _is_1d_vector(const): # scalar basis dot with vector
                data = np.asarray([self.data[0,:] * comp for comp in const])
                return self._with(data, role=self.role)
            elif self.data.shape[0] == const.shape[0]:
                data = np.einsum("kn,k->n", self.data, const, optimize=True)
                if self.is_rhs:
                    return self._with(data, role=f"{self.role}_n") # scalar basis on rhs
                return self._with(data[np.newaxis,:], role=self.role) # scalar basis on lhs
        elif self.role in {"mixed"}:
            if self.shape[0] == const.shape[0]:
                data = np.einsum("kmn,k->mn", self.data, const, optimize=True)
                return self._with(data, role=self.role)
        elif self.role == "scalar":
            data = self.data * const
            role = "vector" if data.ndim == 1 else "scalar"
            return self._with(data, role=role)
        elif self.role == "vector":
            if _is_1d_vector(const):
                data = np.dot(self.data, const)
                role = "scalar" if data.ndim == 0 else "vector"
                return self._with(data, role=role)
            else:
                data = self.data * const 
                role = "vector" if data.ndim == 1 else "scalar"
                return self._with(data, role=role)
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
        if self.role in {"function", "vector"} and grad.role in {"trial", "test"}:
            # Case:  Function · Grad(Trial)      u_k · ∇u_trial
            # (1)  value of u_k at this quad-point
            if self.role == "function":
                u_val = _collapsed_function(self)  # shape (k,)  —   u_k(ξ)
            else:
                u_val = np.asarray(self.data, dtype=float)
            if _is_scalar_field(grad):
                grad_basis = _scalar_grad_basis_matrix(grad)  # (d, n)
                meta = _resolve_meta(self.meta(), meta_grad, prefer='b')
                if _is_1d_vector(u_val) and u_val.shape[0] == grad_basis.shape[0]:
                    data = np.einsum("d,dn->n", u_val, grad_basis, optimize=True)
                    return VecOpInfo(data[np.newaxis, :], role=grad.role, **self.update_meta(meta))
                if u_val.shape[0] == 1:
                    data = u_val[0] * grad_basis
                    return GradOpInfo(data, role=grad.role, **self.update_meta(meta))
                raise NotImplementedError(self._error_msg(grad, "dot_grad"))
            if _is_1d_vector(u_val):
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

        elif self.role in {"trial", "test", "vector"} and grad.role == "function":
            # Case:  Trial · Grad(Function)      u_trial · ∇u_k
            # (1)  value of u_trial at this quad-point
            grad_val = _collapsed_grad(grad)  # shape (k, d)  —   ∇u_k(ξ)
            # (2)  w_i,n = Σ_d ∂_d φ_{k,n} u_d
            lhs_vals = np.asarray(self.data, dtype=float)
            if self.role == "vector":
                if grad_val.shape[0] == 1 and lhs_vals.shape[0] == grad_val.shape[1]:
                    data = float(np.dot(lhs_vals, grad_val[0]))
                    meta = _resolve_meta(self.meta(), meta_grad, prefer='b')
                    return VecOpInfo(np.asarray(data), role="scalar", **self.update_meta(meta))
                if lhs_vals.shape[0] == grad_val.shape[0]:
                    data = np.einsum("s,sd->d", lhs_vals, grad_val, optimize=True)
                    meta = _resolve_meta(self.meta(), meta_grad, prefer='b')
                    return VecOpInfo(data, role="vector", **self.update_meta(meta))
                raise NotImplementedError(self._error_msg(grad, "dot_grad"))
            if grad_val.shape[0] == 1 and lhs_vals.shape[0] == grad_val.shape[1]:
                data = np.einsum("dn,d->n", lhs_vals, grad_val[0], optimize=True)
                meta = _resolve_meta(self.meta(), meta_grad, prefer='a')
                return VecOpInfo(data[np.newaxis, :], role=self.role, **self.update_meta(meta))
            if lhs_vals.shape[0] == grad_val.shape[0]:
                data = np.einsum("sl,sd->dl", lhs_vals, grad_val, optimize=True)
                meta = _resolve_meta(self.meta(), meta_grad, prefer='a')
                return VecOpInfo(data, role=self.role, **self.update_meta(meta))
            else: raise NotImplementedError(self._error_msg(grad, "dot_grad"))
        elif self.role in {"function", "vector"} and grad.role == "function":
            # Case:  Function · Grad(Function)      u_k · ∇u_k
            # (1)  value of u_k at this quad-point
            if self.role == "function":
                u_val = _collapsed_function(self)  # shape (k,)  —   u_k(ξ)
            else:
                u_val = np.asarray(self.data, dtype=float)
            grad_val = _collapsed_grad(grad)  # shape (k, d)  —   ∇u_k(ξ)
            meta = _resolve_meta(self.meta(), meta_grad)
            if grad_val.shape[0] == 1 and u_val.shape[0] == grad_val.shape[1]:
                data = float(np.dot(u_val, grad_val[0]))
                return VecOpInfo(np.asarray(data), role="scalar", **self.update_meta(meta))
            if u_val.shape[0] == grad_val.shape[0]:
                data = np.einsum("s,sd->d", u_val, grad_val, optimize=True)
                return VecOpInfo(data, role="vector",
                                **self.update_meta(meta))
            else: raise NotImplementedError(self._error_msg(grad, "dot_grad"))
        elif self.role == "mixed" and grad.role in {"function", "scalar"}:
            grad_val = _collapsed_grad(grad)
            meta = _resolve_meta(self.meta(), meta_grad, prefer='a')
            if grad_val.shape[0] == 1 and self.shape[0] == grad_val.shape[1]:
                data = np.einsum("kmn,k->mn", self.data, grad_val[0], optimize=True)
                return VecOpInfo(data, role=self.role, **self.update_meta(meta))
            raise NotImplementedError(self._error_msg(grad, "dot_grad"))
        elif self.role in {"trial", "test", "trial_n", "test_n"} and grad.role in {"trial", "test", "trial_n", "test_n"}:
            # Use the gradient-side contraction as the single source of truth
            # for mixed basis vector · basis gradient. This keeps the carried
            # result in canonical test, trial order and avoids duplicating the
            # mixed-basis layout logic in two helpers.
            return grad.left_dot(self)
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

        if self.role in {"function", "vector"} and other_vec.role == "mixed":
            return other_vec.dot_vec(self)
        
        # case 1 function dot test
        if  self.role in {"function", "vector"} and other_vec.role in {"test", "trial"}: # rhs time derivative term
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
            # Bilinear mass-like contraction: keep a semantic mixed scalar carrier
            # in canonical (n_test, n_trial) storage. Mixed scalar carriers are
            # rank-0 tensors over two basis axes, not fake rank-1 objects with a
            # leading singleton component axis.
            meta = _resolve_meta(self.meta(), other_vec.meta())
            data = np.einsum("km,kn->mn", other_vec.data, self.data, optimize=True)
            return VecOpInfo(data, role="mixed", **self.update_meta(meta))
        elif self.role == "test" and other_vec.role == "trial":
            meta = _resolve_meta(self.meta(), other_vec.meta())
            data = np.einsum("km,kn->mn", self.data, other_vec.data, optimize=True)
            return VecOpInfo(data, role="mixed", **self.update_meta(meta))
        
        # case 3 trial and function
        if self.role in {"trial", "test"} and other_vec.role in {"function", "vector"}:
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
        if self.role == "mixed" and other_vec.role == "function":
            meta = _resolve_meta(self.meta(), other_vec.meta(), prefer='a')
            v_values = _collapsed_function(other_vec)  # shape (k,)
            # (k,m,n) dot (k,) -> (m,n)
            if self.shape[0] == v_values.shape[0] and v_values.ndim == 1:
                data = np.einsum("kmn,k->mn", self.data, v_values, optimize=True)
                return VecOpInfo(data, role=self.role, **self.update_meta(meta))
            else:
                raise NotImplementedError(self._error_msg(other_vec, "dot_vec"))
        if self.role == "mixed" and other_vec.role == "vector":
            meta = _resolve_meta(self.meta(), other_vec.meta(), prefer='a')
            v_values = np.asarray(other_vec.data, dtype=float).reshape(-1)
            if self.shape[0] == v_values.shape[0]:
                data = np.einsum("kmn,k->mn", self.data, v_values, optimize=True)
                return VecOpInfo(data, role=self.role, **self.update_meta(meta))
            raise NotImplementedError(self._error_msg(other_vec, "dot_vec"))
        # case 4 function and function
        if self.role == "function" and other_vec.role == "function":
            u_values = _collapsed_function(self)  # shape (k,)
            v_values = _collapsed_function(other_vec)  # shape (k,)
            if u_values.ndim == 1 and v_values.ndim == 1 :
                meta = _resolve_meta(self.meta(), other_vec.meta())
                data = np.dot(u_values, v_values)
                role = "scalar" if data.ndim == 0 else "vector"
                return VecOpInfo(data, role=role, **self.update_meta(meta))
            else:
                raise NotImplementedError(self._error_msg(other_vec, "dot_vec"))
        raise NotImplementedError(f"VecOpInfo.dot_vec not implemented for roles {self.role} and {other_vec.role}."
                                  f" Shapes: {self.data.shape} and {other_vec.data.shape}."
                                  f" Types : {type(self)} and {type(other_vec)}.")
    # ========================================================================
    # Shape, len, and ndim methods
    def __len__(self) -> int:
        """Returns the size of the first dimension (number of components)."""
        return self.data.shape[0] if self.data.ndim > 0 else 1
    def is_scalar_function(self) -> bool:
        """Check if the VecOpInfo represents a scalar function."""
        if self.role in {"function", "scalar"}:
            vals = _collapsed_function(self)  # shape (k,)
            return vals.shape[0] == 1
        elif self.role in {"trial", "test", "trial_n", "test_n"}:
            return self.data.shape[0] == 1
        return False

    def __mul__(self, other: Union[float, np.ndarray]) -> "VecOpInfo":
        """Element-wise multiplication with a scalar or vector."""
        # shape_a = self.data.shape
        # shape_b = getattr(other, 'shape', "None")
        # role_b = getattr(other, 'role', "None")
        # print(f"VecOpInfo.__mul__: roles=({self.role}, {role_b}), shapes=({shape_a}, {shape_b})")
        if isinstance(other, (float, int)):
            return self._with(self.data * other, role=self.role)
        elif isinstance(other, np.ndarray): # np.array
            if other.ndim == 0:
                # Scalar multiplication
                return self._with(self.data * other, role=self.role)
            if self.data.ndim == 0 and other.ndim == 1:
                # Scalar (QP) value times a constant vector -> vector (QP).
                return self._with(self.data * other, role="vector")
            elif other.ndim == 1 and other.size == self.data.shape[0]:
                return self._with(self.data * other[:, np.newaxis], role=self.role)
            elif other.ndim == 1 and self.data.shape[0] == 1:
                # New Case: Scalar multiplication with a vector
                if self.role == "function":
                    vals = _collapsed_function(self)  # shape (k,)
                    return self._with(vals * other, role="vector")
                return self._with(np.asarray([self.data[0,:] * comp for comp in other]), role=self.role)
            elif other.ndim == 2:
                if self.role in {"function", "vector", "trial", "test", "trial_n", "test_n", "mixed"}:
                    try:
                        return self._matrix_contract_right(other)
                    except NotImplementedError:
                        pass
                # Scale a matrix by a scalar-valued function (e.g. div(u_k) * dot(u_trial, v_test)
                # in the skew-symmetric convection Jacobian).
                if self.data.ndim == 0:
                    return other * float(self.data)
                if self.data.ndim == 1 and self.data.shape[0] == 1:
                    return other * float(self.data[0])
                if self.role in {"function", "vector"}:
                    vals = _collapsed_function(self) if self.role == "function" else np.asarray(self.data, dtype=float)
                    vals = np.asarray(vals, dtype=float)
                    if vals.ndim == 1 and other.shape[0] == vals.shape[0]:
                        data = vals @ other
                        role = "scalar" if np.ndim(data) == 0 or (np.ndim(data) == 1 and np.asarray(data).shape == (1,)) else "vector"
                        return VecOpInfo(np.asarray(data), role=role, **self.update_meta(self.meta()))
                if self.role in {"trial","test"} and self.shape[0] == 1 and other.shape == (2,2):
                    # Case: Trial/Test * with identity matrix the result will be GradOpInfo
                    n = self.data.shape[1]
                    k = other.shape[1]
                    res = np.zeros((k,n,k), dtype=self.data.dtype)
                    for i in range(k):
                        for j in range(k):
                            res[i,:,j] = self.data[0,:] * other[i,j]
                    grad_obj = GradOpInfo(res, role=self.role,
                                          field_names=self.field_names,
                                          parent_name=self.parent_name, side=self.side,
                                          field_sides=self.field_sides, is_rhs=self.is_rhs)
                    return grad_obj
                elif self.role in {"mixed"} and other.shape == (2,2) and (
                    self.data.ndim == 2 or (self.data.ndim == 3 and self.shape[0] == 1)
                ):
                    # Mixed scalar carrier * matrix -> mixed rank-2 basis tensor.
                    base = self.data if self.data.ndim == 2 else self.data[0]
                    m, n = base.shape
                    k,d = other.shape
                    res = np.zeros((k,m,n,d), dtype=self.data.dtype)
                    for i in range(k):
                        for j in range(d):
                            res[i,:, :,j] = base * other[i,j]
                    grad_obj = GradOpInfo(res, role=self.role,
                                          field_names=self.field_names,
                                          parent_name=self.parent_name, side=self.side,
                                          field_sides=self.field_sides, is_rhs=self.is_rhs)
                    return grad_obj
                    
            else:
                raise ValueError(f"Cannot multiply VecOpInfo {self.data.shape} with array of shape {other.shape}."
                                 f" Roles: {self.role}, other={getattr(other, 'role', None)}.")
        elif isinstance(other, VecOpInfo):
            # if self.data.shape != other.data.shape:
            #     raise ValueError("VecOpInfo shapes mismatch in multiplication.")
            if self.role == "scalar" and other.role == "scalar":
                data = float(self.data) * float(other.data)
                meta = _resolve_meta(self.meta(), other.meta())
                return VecOpInfo(data, role="scalar", **self.update_meta(meta))
            if self.role == "trial" and other.role == "test":
                # Case: Trial * Test , outer product case
                return np.einsum("km,kn->mn", other.data , self.data, optimize=True)
            elif self.role == "test" and other.role == "trial":
                # Case: Test * Trial , outer product case
                return np.einsum("km,kn->mn", self.data, other.data, optimize=True)
            elif self.role == "function" and other.role == "function":
                # Case: Function * Function
                #
                # - scalar * scalar: pointwise product (treated as dot of 1-vectors)
                # - scalar * vector: component-wise scaling
                # - vector * scalar: component-wise scaling
                # - vector * vector: pointwise dot product (scalar)
                u_vals = _collapsed_function(self)   # shape (k_a,)
                v_vals = _collapsed_function(other)  # shape (k_b,)
                meta = _resolve_meta(self.meta(), other.meta())

                if u_vals.shape[0] == 1 and v_vals.shape[0] > 1:
                    data = float(u_vals[0]) * np.asarray(other.data)
                    return VecOpInfo(data, role="function", **other.update_meta(meta))
                if v_vals.shape[0] == 1 and u_vals.shape[0] > 1:
                    data = float(v_vals[0]) * np.asarray(self.data)
                    return VecOpInfo(data, role="function", **self.update_meta(meta))

                data = np.dot(u_vals, v_vals)
                role = "scalar" if np.ndim(data) == 0 else "vector"
                return VecOpInfo(data, role=role, **self.update_meta(meta))
            elif self.role == "scalar" and other.role != "scalar":
                # Case: scalar coefficient (collapsed) scaling a basis/vector/function
                scale = float(self.data)
                meta = _resolve_meta(self.meta(), other.meta(), prefer="b")
                return VecOpInfo(scale * other.data, role=other.role, **other.update_meta(meta))
            elif self.role != "scalar" and other.role == "scalar":
                # Case: basis/vector/function scaled by a scalar coefficient on the right
                scale = float(other.data)
                meta = _resolve_meta(self.meta(), other.meta(), prefer="a")
                return VecOpInfo(scale * self.data, role=self.role, **self.update_meta(meta))
            elif self.role in {"trial", "test"} and other.role in {"function", "vector"} and other.is_scalar_function():
                # Scalar coefficient function/value scaling a vector-valued basis carrier.
                scale = float(_collapsed_function(other)[0])
                meta = _resolve_meta(self.meta(), other.meta(), prefer="a")
                return VecOpInfo(scale * self.data, role=self.role, **self.update_meta(meta))
            elif self.role in {"function", "vector"} and other.role in {"trial", "test"} and self.is_scalar_function():
                # Scalar coefficient function/value scaling a vector-valued basis carrier.
                scale = float(_collapsed_function(self)[0])
                meta = _resolve_meta(self.meta(), other.meta(), prefer="b")
                return VecOpInfo(scale * other.data, role=other.role, **other.update_meta(meta))
            elif self.role in {"trial","test"} and other.role in {"function", "vector"}:
                # Case: Scalar Trial/Test * Function 
                v_vals = _collapsed_function(other)  # shape (k,)
                if self.shape[0] == 1 and self.shape[1] >1 and v_vals.shape[0] > 1:
                    # Special case: single component trial/test function
                    data = np.asarray([self.data[0,:] * comp for comp in v_vals])
                    meta = _resolve_meta(self.meta(), other.meta(), prefer='a')
                    return VecOpInfo(data, role=self.role, **self.update_meta(meta))
                elif self.shape[0] == 1 and self.shape[1] >1 and v_vals.shape[0] == 1:
                    # Special case: single component function
                    data = v_vals[0] * self.data
                    meta = _resolve_meta(self.meta(), other.meta(), prefer='a')
                    return VecOpInfo(data, role=self.role, **self.update_meta(meta))
                else:
                    raise NotImplementedError(self._error_msg(other, "VecOpInfo.__mul__"))
            elif self.role in {"function", "vector"} and other.role in {"trial","test"}:
                # Case: Function * Scalar Trial/Test
                u_vals = _collapsed_function(self)  # shape (k,)
                if other.shape[0] == 1 and other.shape[1] >1 and u_vals.shape[0] > 1:
                    # Special case: single component trial/test function
                    data = np.asarray([other.data[0,:] * comp for comp in u_vals])
                    meta = _resolve_meta(self.meta(), other.meta(), prefer='b')
                    return VecOpInfo(data, role=other.role, **self.update_meta(meta))
                elif other.shape[0] == 1 and other.shape[1] >1 and u_vals.shape[0] == 1:
                    # Special case: single component function
                    data = u_vals[0] * other.data
                    meta = _resolve_meta(self.meta(), other.meta(), prefer='b')
                    return VecOpInfo(data, role=other.role, **self.update_meta(meta))
                else:
                    raise NotImplementedError(self._error_msg(other, "VecOpInfo.__mul__"))
            elif self.role == "function" and other.role == "vector" and self.is_scalar_function():
                # Case: scalar coefficient function scaling a pre-expanded local RHS vector.
                scale = float(_collapsed_function(self)[0])
                return other._with(scale * other.data, role=other.role)
            elif self.role == "vector" and other.role == "function" and other.is_scalar_function():
                scale = float(_collapsed_function(other)[0])
                return self._with(scale * self.data, role=self.role)
            # elif self.role in {"trial","test"} and other.role == "function":
            #     # Case: Trial * Function , dot product case
            #     meta = _resolve_meta(self.meta(), other.meta(), prefer='a')
            #     v_vals = _collapsed_function(other)  # shape (k,)
            #     data = np.einsum("kn,k->n", self.data, v_vals, optimize=True)
            #     return self._expand_axis_lhs(data, self.role, meta)
            # elif self.role == "function" and other.role in {"trial", "test"}:
            #     meta = _resolve_meta(self.meta(), other.meta(), prefer='b')
            #     # Case: Function * Trial , dot product case
            #     u_vals = _collapsed_function(self)  # shape (k,)
            #     data = np.einsum("k,kn->n", u_vals, other.data, optimize=True)
            #     return self._expand_axis_lhs(data, other.role, meta)
            elif self.role == "function" and other.role == "mixed":
                # Scalar function scaling a mixed block (e.g., shape sensitivities)
                u_vals = _collapsed_function(self)  # (k,)
                if u_vals.shape[0] not in (1, other.data.shape[0]):
                    raise ValueError(f"Cannot scale mixed VecOpInfo of shape {other.data.shape} with function of shape {u_vals.shape}.")
                # allow either per-component scaling or a single scalar
                if u_vals.shape[0] == other.data.shape[0]:
                    scale = u_vals.reshape((u_vals.shape[0],) + (1,) * (other.data.ndim - 1))
                else:
                    scale = u_vals[0]
                data = other.data * scale
                meta = _resolve_meta(self.meta(), other.meta(), prefer='b')
                return VecOpInfo(data, role=other.role, **self.update_meta(meta))
            elif self.role == "mixed" and other.role == "function":
                # Mixed block scaled by a scalar function on the right
                v_vals = _collapsed_function(other)  # (k,)
                if v_vals.shape[0] not in (1, self.data.shape[0]):
                    raise ValueError(f"Cannot scale mixed VecOpInfo of shape {self.data.shape} with function of shape {v_vals.shape}.")
                if v_vals.shape[0] == self.data.shape[0]:
                    scale = v_vals.reshape((v_vals.shape[0],) + (1,) * (self.data.ndim - 1))
                else:
                    scale = v_vals[0]
                data = self.data * scale
                meta = _resolve_meta(self.meta(), other.meta(), prefer='a')
                return VecOpInfo(data, role=self.role, **self.update_meta(meta))
            elif self.role == "function" and other.role in {"trial_n", "test_n"} and self.is_scalar_function():
                # Scalar function scaling a RHS basis vector (e.g. div(u_k) * dot(u_k, v_test))
                scale = float(_collapsed_function(self)[0])
                meta = _resolve_meta(self.meta(), other.meta(), prefer="b")
                return VecOpInfo(scale * other.data, role=other.role, **self.update_meta(meta))
            elif self.role in {"trial_n", "test_n"} and other.role == "function" and other.is_scalar_function():
                scale = float(_collapsed_function(other)[0])
                meta = _resolve_meta(self.meta(), other.meta(), prefer="a")
                return VecOpInfo(scale * self.data, role=self.role, **self.update_meta(meta))
            else:
                raise NotImplementedError(f"VecOpInfo multiplication not implemented for roles {self.role} and {other.role}."
                                          f" (self.shape={self.shape}, other.shape={other.shape})")
                
        elif isinstance(other, GradOpInfo):
            if self.role == "scalar":
                # Scalar coefficient/value scaling a gradient table.
                s = np.asarray(self.data)
                if s.ndim == 0:
                    scale = float(s)
                elif s.ndim == 1 and s.shape[0] == 1:
                    scale = float(s[0])
                else:
                    raise NotImplementedError(
                        f"Scaling GradOpInfo with non-scalar VecOpInfo(role='scalar', shape={s.shape}) is not supported."
                    )
                return other._with(scale * other.data, role=other.role, coeffs=other.coeffs)
            # NOTE: some scalar coefficient products collapse to role="scalar" with shape=().
            # Avoid indexing self.shape[0] before checking the relevant roles.
            if self.role in {"trial", "test"} and other.shape == (2, 2) and self.shape[:1] == (1,):
                # Case: scalar trial or test with identity matrix the result will be GradOpInfo
                # print(f"self.shape: {self.shape}, other.shape: {other.shape}"
                #       f" with roles {self.role} and {other.role}")
                n = self.data.shape[1]
                k = other.shape[1]
                res = np.zeros((k,n,k), dtype=self.data.dtype)
                for i in range(k):
                    for j in range(k):
                        res[i,:,j] = self.data[0,:] * other.data[i,j]
                grad_obj = GradOpInfo(res, role=self.role,
                                      field_names=self.field_names,
                                      parent_name=self.parent_name, side=self.side,
                                      field_sides=self.field_sides, is_rhs=self.is_rhs)
                return grad_obj
            elif self.role == "function" and other.role in {"function", "identity"} and other.shape==(2,2):
                # Case: function dot with identity matrix
                u_vals = _collapsed_function(self)  # shape (k,)
                data = np.zeros((2,2))  # assuming 2D
                if u_vals.ndim ==1 and u_vals.shape[0] == 1:
                    val = u_vals[0]
                    data += val * other.data
                else:
                    for i in range(2):
                        for j in range(2):
                            data[i,j] += u_vals[i] * other.data[i,j]
                role = "function"
                meta = _resolve_meta(self.meta(), other.meta(), prefer='a')
                return GradOpInfo(data, role=role, **self.update_meta(meta))
            elif self.role == "function" and other.role == "identity":
                u_vals = _collapsed_function(self)  # shape (k,)
                if u_vals.shape[0] == 1 and u_vals.ndim == 1:
                    data = u_vals[0] * other.data
                else:
                    raise NotImplementedError("Function dot with identity only implemented for single-component functions.")
                role = "function"
                meta = _resolve_meta(self.meta(), other.meta(), prefer='a')
                return GradOpInfo(data, role=role, **self.update_meta(meta))
            elif self.role == "function" and self.is_scalar_function() and other.role == "function":
                # Scalar coefficient function times Grad(Function) stays gradient-valued.
                u_vals = _collapsed_function(self)  # shape (1,)
                meta = _resolve_meta(self.meta(), other.meta(), prefer="a")
                data = u_vals[0] * _collapsed_grad(other)
                return GradOpInfo(data, role="function", **self.update_meta(meta))
            elif (
                self.role == "mixed"
                and other.role in {"function", "identity"}
                and other.shape == (2, 2)
                and (
                    self.data.ndim == 2
                    or (self.data.ndim == 3 and self.data.shape[0] == 1)
                )
            ):
                # Mixed scalar carrier * matrix -> mixed rank-2 tensor.
                base = self.data if self.data.ndim == 2 else self.data[0]
                m, n = base.shape
                k,d = other.shape
                data = np.zeros((k,m,n,d), dtype=self.data.dtype)
                for i in range(k):
                    for j in range(d):
                        data[i, :, :, j] = base * other.data[i, j]
                role = "mixed"
                meta = _resolve_meta(self.meta(), other.meta(), prefer='a')
                return GradOpInfo(data, role=role, **self.update_meta(meta))
            elif self.role == "trial" and other.role == "test" and self.shape[0]==1:
                # Case: Scalar Trial * Grad(Test)
                # (1,n) * (k, n, d) -> (k, n, d)
                if self.shape[0] != 1:
                    raise NotImplementedError("Only scalar trial factors supported here.")
                meta = _resolve_meta(other.meta(), self.meta(), prefer="b")
                # trial_vals: (1, 1, n_trial, 1); test_grad: (k, n_test, 1, d)
                n_trial = self.data.shape[1]
                n_test  = other.data.shape[1]
                k       = other.data.shape[0]
                d       = other.data.shape[2]

                res = np.zeros((k, n_test, n_trial, d), dtype=self.data.dtype)
                for i in range(k):
                    for j in range(d):
                        # outer product: test rows × trial columns
                        res[i, :, :, j] = np.outer(other.data[i, :, j], self.data[0, :])
                return GradOpInfo(res, role="mixed", **self.update_meta(meta))

            elif self.role == "test" and other.role == "trial" and self.shape[0]==1:
                if self.shape[0] != 1:
                    raise NotImplementedError("Only scalar test factors supported here.")
                meta = _resolve_meta(self.meta(), other.meta(), prefer="a")
                n_test = self.data.shape[1]
                n_trial = other.data.shape[1]
                k = other.data.shape[0]
                d = other.data.shape[2]
                res = np.zeros((k, n_test, n_trial, d), dtype=self.data.dtype)
                for i in range(k):
                    for j in range(d):
                        res[i, :, :, j] = np.outer(self.data[0, :], other.data[i, :, j])
                return GradOpInfo(res, role="mixed", **self.update_meta(meta))
            elif self.role == "function" and other.role in {"trial", "test", "mixed"} and self.is_scalar_function():
                # Case: Scalar Function * Grad(Trial/Test)
                u_vals = _collapsed_function(self)  # shape (k,)
                if u_vals.shape[0] != 1:
                    raise NotImplementedError("Only scalar function factors supported here.")
                meta = _resolve_meta(self.meta(), other.meta(), prefer="b")
                data = u_vals[0] * other.data
                return GradOpInfo(data, role=other.role, **self.update_meta(meta))
            elif self.role in {"trial", "test"} and _is_scalar_field(self) and other.role == "function":
                # Scalar Trial/Test * Grad(Function) is a scalar-times-vector product,
                # so it stays gradient-valued with the basis role.
                grad_vals = _collapsed_grad(other)  # shape (k, d)
                meta = _resolve_meta(self.meta(), other.meta(), prefer="a")
                if grad_vals.shape[0] != 1:
                    raise NotImplementedError("Only scalar gradient values are supported in scalar basis promotion.")
                basis_vals = np.asarray(self.data)
                if basis_vals.ndim == 2:
                    basis_vals = basis_vals[0]
                data = np.ascontiguousarray(grad_vals[0])[:, np.newaxis] * np.ascontiguousarray(basis_vals)[None, :]
                return GradOpInfo(data, role=self.role, **self.update_meta(meta))


            else:
                raise NotImplementedError(f"VecOpInfo multiplication not implemented for roles {self.role} and GradOpInfo role {other.role}."
                                          f" (self.shape={self.shape}, other.shape={other.shape})")
        else:
            role_other = getattr(other, 'role', None)
            shape_other = getattr(other, 'shape', None)
            raise TypeError(f"Unsupported multiplication type: {type(other)}"
                            f" for VecOpInfo with shape {self.data.shape} and role {self.role}."
                            f" Other role: {role_other}, shape: {shape_other}.")
    def __rmul__(self, other: Union[float, np.ndarray]) -> "VecOpInfo":
        if isinstance(other, np.ndarray):
            other = np.asarray(other)
            if other.ndim == 2 and self.role in {"function", "vector", "trial", "test", "trial_n", "test_n", "mixed"}:
                try:
                    return self._matrix_contract_left(other)
                except NotImplementedError:
                    pass
        return self.__mul__(other)

    def __truediv__(self, other):
        """Element-wise division by a scalar (float/int/0-d array)."""
        if isinstance(other, (float, int, np.floating)):
            return self._with(self.data / other, role=self.role)
        if isinstance(other, np.ndarray) and other.ndim == 0:
            return self._with(self.data / float(other), role=self.role)
        if isinstance(other, VecOpInfo) and other.role in {"function", "scalar"}:
            # Pointwise division by a scalar coefficient function/value.
            # This is needed for rational nonlinearities such as S/(K+S).
            return self._with(self.data / other.data, role=self.role)
        raise TypeError(f"VecOpInfo can only be divided by a scalar, not {type(other)}")

    def __rtruediv__(self, other):
        """Scalar divided by this VecOpInfo (element-wise)."""
        if isinstance(other, (float, int, np.floating)):
            return self._with(other / self.data, role=self.role)
        if isinstance(other, np.ndarray) and other.ndim == 0:
            return self._with(float(other) / self.data, role=self.role)
        raise TypeError(f"Scalar division only; got numerator of type {type(other)}")

    def __add__(self, other: "VecOpInfo") -> "VecOpInfo":
        """Element-wise addition with another VecOpInfo/np.ndarray."""
        import numbers as _numbers
        if isinstance(other, BaseOpInfo):
            try:
                from pycutfem.ufl.tensor_algebra import TensorRuleEngine

                sum_lowering = TensorRuleEngine.plan_sum_lowering(self, other)
            except Exception:
                sum_lowering = None
            if sum_lowering is not None:
                lhs = _apply_storage_transform_opinfo(self, sum_lowering.lhs_transform)
                rhs = _apply_storage_transform_opinfo(other, sum_lowering.rhs_transform)
                if lhs is not self or rhs is not other:
                    return lhs.__add__(rhs)

        # Allow adding scalar constants (float/int/0d arrays) to vector-like ops.
        if isinstance(other, (_numbers.Number, np.number)):
            val = float(other)
            return VecOpInfo(self.data + val, role=self.role, **self.update_meta(self.meta()))
        if isinstance(other, np.ndarray) and other.ndim == 0:
            val = float(other)
            return VecOpInfo(self.data + val, role=self.role, **self.update_meta(self.meta()))

        other_data = getattr(other, 'data', other)
        other_meta = getattr(other, 'meta', lambda: None)()
        other_type = getattr(other, 'type', None)
        other_role = getattr(other, 'role', None)
        if not isinstance(other, (VecOpInfo, np.ndarray)):
            raise TypeError(f"Cannot add VecOpInfo to {type(other)}."
                            f" with shapes {self.data.shape} and {getattr(other_data, 'shape', ())}")

        # Allow adding a scalar-valued VecOpInfo to a scalar/vector-valued VecOpInfo
        # by treating it like a numeric constant. This occurs naturally for products
        # of scalar coefficient functions which collapse to role="scalar" (shape=()).
        if isinstance(other, VecOpInfo) and other_role == "scalar" and np.asarray(other_data).ndim == 0 and self.role in {"function", "vector", "scalar"}:
            val = float(other_data)
            meta = _resolve_meta(self.meta(), other_meta, strict=False)
            return VecOpInfo(self.data + val, role=self.role, **self.update_meta(meta))

        # Broadcast true scalars (shape=()) into length-1 vectors (shape=(1,)).
        if self.role == "scalar" and np.asarray(self.data).ndim == 0 and isinstance(other, VecOpInfo) and other_role in {"vector", "function"}:
            arr = np.asarray(other_data)
            if arr.ndim == 1 and arr.shape[0] == 1:
                val = float(self.data)
                meta = _resolve_meta(self.meta(), other_meta, strict=False)
                return VecOpInfo(arr + val, role=other_role, **other.update_meta(meta))
        if self.role in {"mixed"} and other_role is None and self.data.shape[0] == 1 and self.ndim == 3:
            if self.shape[1] == other.shape[0] and self.shape[2] == other.shape[1]:
                # Case: mixed + np.ndarray (k,m,n) + (m,n)
                res = self.data[0,:,:] + other_data
                return VecOpInfo(res[np.newaxis,:,:], role=self.role, **self.update_meta(self.meta()))

        if self.role in {"test", "test_n"} and other_role in {"test", "test_n"}:
            # a: (1,n) or (n,) and b: (1,n) or (n,)
            is_a_row_vec = self.data.ndim ==2 and self.data.shape[0]==1
            is_b_row_vec = other_data.ndim ==2 and other_data.shape[0]==1
            if is_a_row_vec and is_b_row_vec:
                data = self.data + other_data
            elif is_a_row_vec and not is_b_row_vec:
                data = self.data[0,:] + other_data
            elif not is_a_row_vec and is_b_row_vec:
                data = self.data + other_data[0,:]
            else:
                data = self.data + other_data
            meta = _resolve_meta(self.meta(), other_meta, prefer='a')
            return VecOpInfo(data, role=self.role, **self.update_meta(meta))
        # Allow adding row-test vectors (1,n) with plain vectors (n,) by broadcasting.
        if self.role in {"test", "test_n"} and np.asarray(self.data).ndim == 2 and self.data.shape[0] == 1:
            arr_other = np.asarray(other_data)
            if arr_other.ndim == 1 and arr_other.shape[0] == self.data.shape[1]:
                meta = _resolve_meta(self.meta(), other_meta, prefer='a')
                return VecOpInfo(self.data[0, :] + arr_other, role=self.role, **self.update_meta(meta))
        if other_role in {"test", "test_n"} and isinstance(other, VecOpInfo):
            arr_self = np.asarray(self.data)
            arr_other = np.asarray(other_data)
            if arr_self.ndim == 1 and arr_other.ndim == 2 and arr_other.shape[0] == 1 and arr_self.shape[0] == arr_other.shape[1]:
                meta = _resolve_meta(self.meta(), other_meta, prefer='b')
                return VecOpInfo(arr_self + arr_other[0, :], role=other_role, **other.update_meta(meta))
        if isinstance(other, VecOpInfo):
            arr_self = np.asarray(self.data)
            arr_other = np.asarray(other_data)
            try:
                bshape = np.broadcast_shapes(arr_self.shape, arr_other.shape)
            except ValueError:
                bshape = None
            if bshape is not None and (bshape != arr_self.shape or bshape != arr_other.shape):
                data = np.broadcast_to(arr_self, bshape) + np.broadcast_to(arr_other, bshape)
                prefer_other = other_role in {"trial", "test", "trial_n", "test_n", "mixed"} and self.role in {"function", "vector", "scalar"}
                if prefer_other:
                    meta = _resolve_meta(self.meta(), other_meta, prefer='b')
                    return VecOpInfo(data, role=other_role, **other.update_meta(meta))
                meta = _resolve_meta(self.meta(), other_meta, prefer='a')
                return VecOpInfo(data, role=self.role, **self.update_meta(meta))
        arr_self = np.asarray(self.data)
        arr_other = np.asarray(other_data)
        if arr_self.shape != arr_other.shape:
            self_is_zero = arr_self.size == 0 or np.count_nonzero(arr_self) == 0
            other_is_zero = arr_other.size == 0 or np.count_nonzero(arr_other) == 0
            if self_is_zero:
                if isinstance(other, VecOpInfo):
                    meta = _resolve_meta(self.meta(), other_meta, prefer='b')
                    return VecOpInfo(arr_other.copy(), role=other_role, **other.update_meta(meta))
                return arr_other.copy()
            if other_is_zero:
                if isinstance(other, VecOpInfo):
                    meta = _resolve_meta(self.meta(), other_meta, prefer='a')
                    return VecOpInfo(arr_self.copy(), role=self.role, **self.update_meta(meta))
                return VecOpInfo(arr_self.copy(), role=self.role, **self.update_meta(self.meta()))
        if self.data.shape != other_data.shape:
            raise ValueError(f"VecOpInfo shapes mismatch in addition: {self.data.shape} vs {other_data.shape}"
                             f" Roles: {self.role}, other={getattr(other, 'role', None)}."
                             f" Types: {type(self)} vs {type(other)}")
        if type(self) == type(other):
            meta = _resolve_meta(self.meta(), other_meta, strict=False)
        elif type(other) == np.ndarray:
            meta = self.meta()
        return VecOpInfo(self.data + other_data, role=self.role, **self.update_meta(meta))
    def __sub__(self, other: "VecOpInfo") -> "VecOpInfo":
        return self.__add__(-other)

    def __radd__(self, other) -> "VecOpInfo":
        return self.__add__(other)

    def __rsub__(self, other) -> "VecOpInfo":
        return (-self).__add__(other)

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
    __array_priority__ = 1000
    coeffs: np.ndarray = field(default=None)  # (k, n) when role == "function" and data.ndim == 3

    def __post_init__(self):
        data = np.asarray(self.data)
        if self.role in {"trial", "test", "trial_n", "test_n"} and data.ndim == 3 and int(data.shape[0]) == 1:
            # Canonical scalar basis gradients as rank-1 spatial tensors carried over basis columns: (d, n).
            object.__setattr__(self, "data", np.ascontiguousarray(data[0].T))

    def _with(self, data: np.ndarray, role: str | None = None, coeffs: np.ndarray | None = None) -> "GradOpInfo":
        return GradOpInfo(
            np.asarray(data),
            role=(role or self.role),
            coeffs=self.coeffs if coeffs is None else coeffs,
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
        tensor_rank = _tensor_rank_of_opinfo(self)
        if tensor_rank is not None and tensor_rank != 2:
            raise ValueError(
                f"Transpose is only defined for rank-2 tensors; GradOpInfo with shape {self.data.shape} "
                f"has semantic tensor rank {tensor_rank}."
            )
        if self.data.ndim == 4:      # (k, n, m, d)
            if self.role == "mixed":
                G = self.data                        # (k, n, m, d)
                k, n, m, d = G.shape
                Gswap = np.empty_like(G)
                # S[i, :, :, j] = G[j, :, :, i]
                for i in range(k):
                    for j in range(d):
                        Gswap[i, :, :, j] = G[j, :, :, i]
                return self._with(Gswap, role=self.role, coeffs=self.coeffs)
        if self.data.ndim == 3:        # (k, n, d)
            if self.role == "function":
                # Transpose for function gradients
                # swap the coefficents to match the new shape
                grad_vals = _collapsed_grad(self)  # (k, d)
                return self._with(grad_vals.T, role=self.role, coeffs=None)
            elif self.role in {"trial", "test", "trial_n", "test_n"}: # trial and test
                G = self.data                        # (k, n, d)
                k, n, d = G.shape
                Gswap = np.empty_like(G)
                # S[i, :, j] = G[j, :, i]
                for i in range(k):
                    for j in range(d):
                        Gswap[i, :, j] = G[j, :, i]
                return self._with(Gswap, role=self.role, coeffs=self.coeffs)
            else:
                raise ValueError(f"Cannot transpose GradOpInfo with role '{self.role}' and data of shape {self.data.shape}.")
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

        raise ValueError(f"Unexpected function gradient shape: {self.data.shape}"
                         f" with role {self.role}.")
    

    def inner(self, other: "GradOpInfo") -> np.ndarray:
        if not isinstance(other, GradOpInfo):
            raise ValueError(f"Inner product requires another GradOpInfo, got {type(other)}."
                f"Or Incompatible GradOpInfo shapes: {self.data.shape} and {other.data.shape}.")
        if self.is_rhs:
            if self.role in {"function", "identity", "scalar"} and other.role in {"trial", "test"}:
                # Case: Function · Grad(Trial) or Grad(Test)  -> (n,)
                kd = _collapsed_grad(self)  # shape (k, d)  —   ∇u_k(ξ)
                if _is_scalar_field(other) and kd.shape[0] == 1:
                    return np.einsum("d,dn->n", kd[0], _scalar_grad_basis_matrix(other), optimize=True)
                if kd.shape[0] == other.shape[0]:
                    return  np.einsum("kd,knd->n", kd, other.data, optimize=True)
                else: raise NotImplementedError(self._error_msg(other, "inner between gradients"))
            elif self.role in {"trial", "test"} and other.role in {"function", "identity", "scalar"}:
                # Case: Grad(Trial) or Grad(Test) · Function  -> (n,)
                kd = _collapsed_grad(other)  # shape (k, d)  —   ∇u_k(ξ)
                if _is_scalar_field(self) and kd.shape[0] == 1:
                    return np.einsum("dn,d->n", _scalar_grad_basis_matrix(self), kd[0], optimize=True)
                if kd.shape[0] == self.shape[0]:
                    return np.einsum("knd,kd->n", self.data, kd, optimize=True)
                else: raise NotImplementedError(self._error_msg(other, "inner between gradients"))
            elif self.role in {"function", "identity", "scalar"} and other.role in {"function", "identity", "scalar"}:
                # Case: Function · Function  -> ()
                if self.role in {"function", "scalar"}:
                    kd_self = _collapsed_grad(self)  # shape (k, d)  —   ∇u_k(ξ)
                else:
                    kd_self = self.data  # shape (k, d)  —   ∇u_k(ξ)
                if other.role in {"function", "scalar"}:
                    kd_other = _collapsed_grad(other)  # shape (k, d)  —   ∇u_k(ξ)
                else:
                    kd_other = other.data  # shape (k, d)  —   ∇u_k(ξ)
                if kd_self.shape[0] == kd_other.shape[0]:
                    return np.einsum("kd,kd->", kd_self, kd_other, optimize=True)
                else: raise NotImplementedError(self._error_msg(other, "inner between gradients"))
            else:
                raise NotImplementedError(self._error_msg(other, "inner between gradients"))

        if self.role == "test" and other.role == "trial":
            # standard order: rows=test, cols=trial
            if _is_scalar_field(self) and _is_scalar_field(other):
                return np.einsum("dn,dm->nm", _scalar_grad_basis_matrix(self), _scalar_grad_basis_matrix(other), optimize=True)
            return np.einsum("knd,kmd->nm", self.data, other.data, optimize=True)

        elif self.role == "trial" and other.role == "test":
            # reversed inputs: build rows=test, cols=trial anyway
            if _is_scalar_field(self) and _is_scalar_field(other):
                return np.einsum("dn,dm->mn", _scalar_grad_basis_matrix(self), _scalar_grad_basis_matrix(other), optimize=True)
            return np.einsum("knd,kmd->mn", self.data, other.data, optimize=True)
        elif self.role in {"function", "identity", "scalar"} and other.role in {"trial", "test"}:
            # Case: Function · Grad(Trial) or Grad(Test)  -> (n,)
            kd = _collapsed_grad(self)  # shape (k, d)  —   ∇u_k(ξ)
            if _is_scalar_field(other) and kd.shape[0] == 1:
                return np.einsum("d,dn->n", kd[0], _scalar_grad_basis_matrix(other), optimize=True)
            if kd.shape[0] == other.shape[0]:
                return np.einsum("kd,knd->n", kd, other.data, optimize=True)
            else: raise NotImplementedError(self._error_msg(other, "inner between gradients"))
        elif self.role in {"trial", "test"} and other.role in {"function", "identity", "scalar"}:
            # Case: Grad(Trial) or Grad(Test) · Function  -> (n,)
            kd = _collapsed_grad(other)  # shape (k, d)  —   ∇u_k(ξ)
            if _is_scalar_field(self) and kd.shape[0] == 1:
                return np.einsum("dn,d->n", _scalar_grad_basis_matrix(self), kd[0], optimize=True)
            if kd.shape[0] == self.shape[0]:
                return np.einsum("knd,kd->n", self.data, kd, optimize=True)
            else: raise NotImplementedError(self._error_msg(other, "inner between gradients"))
        elif self.role in {"function", "identity", "scalar"} and other.role in {"mixed"}:
            # Case: Function · Grad(Mixed)  -> (n,)
            kd = _collapsed_grad(self)  # shape (k, d)  —   ∇u_k(ξ)
            if kd.shape[0] == other.shape[0]:
                return np.einsum("kd,knmd->nm", kd, other.data, optimize=True)
            else: raise NotImplementedError(self._error_msg(other, "inner between gradients"))
        elif self.role in {"mixed"} and other.role in {"function", "identity", "scalar"}:
            # Case: Grad(Mixed) · Function  -> (n,)
            kd = _collapsed_grad(other)  # shape (k, d)  —   ∇u_k(ξ)
            if kd.shape[0] == self.shape[0]:
                return np.einsum("knmd,kd->nm", self.data, kd, optimize=True)
            else: raise NotImplementedError(self._error_msg(other, "inner between gradients"))
        elif self.role in {"function", "identity", "scalar"} and other.role in {"function", "identity", "scalar"}:
            # (RHS or unusual cases rarely hit here; keep the default if needed)
            if self.role in {"function", "scalar"}:
                kd_self = _collapsed_grad(self)  # shape (k, d)  —   ∇u_k(ξ)
            else:
                kd_self = self.data  # shape (k, d)  —   ∇u_k(ξ)
            if other.role in {"function", "scalar"}:
                kd_other = _collapsed_grad(other)  # shape (k, d)  —   ∇u_k(ξ)
            else:
                kd_other = other.data  # shape (k, d)  —   ∇u_k(ξ)
            if kd_self.shape[0] == kd_other.shape[0]:
                return np.einsum("kd,kd->", kd_self, kd_other, optimize=True)
            else: raise NotImplementedError(self._error_msg(other, "inner between gradients"))
            # return np.einsum("knd,kmd->nm", self.data, other.data, optimize=True)
        else:
            raise NotImplementedError(f"GradOpInfo.inner not implemented for roles {self.role} and {other.role}."
                                      f" Shapes: {self.data.shape} and {other.data.shape}."
                                      f" Roles: {self.role} and {other.role}.")


    def dot(self, other:"GradOpInfo") -> "BaseOpInfo":
        """
        Computes dot(∇u, ∇v) for two GradOpInfo objects.

        Depending on the surviving free axes, the result may still be a
        gradient/matrix carrier or may collapse to a vector/scalar carrier.
        """
        if not isinstance(other, GradOpInfo):
            raise TypeError(f"Expected GradOpInfo, got {type(other)}.")

        def _value_role_from_vec(data: np.ndarray) -> str:
            arr = np.asarray(data)
            if arr.ndim == 0:
                return "scalar"
            if arr.ndim == 1 and arr.shape[0] == 1:
                return "scalar"
            return "vector"

        # Case 1: value-gradient · basis-gradient
        if self.role in {"function", "scalar"} and other.role in {"trial", "test"}:
            grad_val = _collapsed_grad(self)  # shape (k, d)
            meta = _resolve_meta(self.meta(), other.meta(), prefer='b')
            if _is_scalar_field(other):
                g_basis = _scalar_grad_basis_matrix(other)  # (d, n)
                if grad_val.shape[1] == g_basis.shape[0]:
                    data = grad_val @ g_basis
                    return VecOpInfo(data, role=other.role, **self.update_meta(meta))
                raise NotImplementedError(self._error_msg(other, "dot between gradients"))
            if grad_val.shape[0] == 1 and other.shape[-1] == grad_val.shape[1]:
                data = np.einsum("s,snd->dn", grad_val[0], other.data, optimize=True)
                return VecOpInfo(data, role=other.role, **self.update_meta(meta))
            if grad_val.shape[-1] == other.shape[0]:
                data = np.einsum("ks,snd->knd", grad_val, other.data, optimize=True)
                return GradOpInfo(data, role=other.role, **self.update_meta(meta))
            raise NotImplementedError(self._error_msg(other, "dot between gradients"))

        # Case 2: basis-gradient · value-gradient
        if self.role in {"trial", "test"} and other.role in {"function", "scalar"}:
            grad_val = _collapsed_grad(other)  # shape (k, d)
            meta = _resolve_meta(self.meta(), other.meta(), prefer='a')
            if _is_scalar_field(self):
                g_basis = _scalar_grad_basis_matrix(self)  # (d, n)
                if grad_val.shape[0] == 1 and g_basis.shape[0] == grad_val.shape[1]:
                    data = grad_val[0] @ g_basis
                    return VecOpInfo(data[np.newaxis, :], role=self.role, **self.update_meta(meta))
                if g_basis.shape[0] == grad_val.shape[0]:
                    data = grad_val.T @ g_basis
                    return VecOpInfo(data, role=self.role, **self.update_meta(meta))
                raise NotImplementedError(self._error_msg(other, "dot between gradients"))
            if grad_val.shape[0] == 1 and self.shape[-1] == grad_val.shape[1]:
                data = np.einsum("knd,d->kn", self.data, grad_val[0], optimize=True)
                return VecOpInfo(data, role=self.role, **self.update_meta(meta))
            if self.shape[-1] == grad_val.shape[0]:
                data = np.einsum("kns,sd->knd", self.data, grad_val, optimize=True)
                return GradOpInfo(data, role=self.role, **self.update_meta(meta))
            raise NotImplementedError(self._error_msg(other, "dot between gradients"))

        if self.role in {"mixed"} and other.role in {"function", "scalar"}:
            other_grad_val = _collapsed_grad(other)  # shape (k, d)
            if self.shape[-1] != other_grad_val.shape[0]:
                raise NotImplementedError(self._error_msg(other, "dot between gradients"))
            data = np.einsum("knms,sd->knmd", self.data, other_grad_val, optimize=True)
            meta = _resolve_meta(self.meta(), other.meta(), prefer='a')
            return GradOpInfo(data, role=self.role, **self.update_meta(meta))

        if self.role in {"function", "scalar"} and other.role in {"mixed"}:
            self_grad_val = _collapsed_grad(self)  # shape (k, d)
            if self_grad_val.shape[-1] != other.shape[0]:
                raise NotImplementedError(self._error_msg(other, "dot between gradients"))
            data = np.einsum("ks,snmd->knmd", self_grad_val, other.data, optimize=True)
            meta = _resolve_meta(self.meta(), other.meta(), prefer='b')
            return GradOpInfo(data, role=other.role, **self.update_meta(meta))

        # Case 3: value-gradient · value-gradient
        if self.role in {"function", "identity", "scalar"} and other.role in {"function", "identity", "scalar"}:
            grad_val = _collapsed_grad(self)
            other_grad_val = _collapsed_grad(other)
            meta = _resolve_meta(self.meta(), other.meta())
            if grad_val.shape[1] == other_grad_val.shape[0]:
                data = grad_val @ other_grad_val
                if np.ndim(data) == 0 or (isinstance(data, np.ndarray) and data.shape == (1, 1)):
                    return VecOpInfo(np.asarray(float(np.asarray(data).reshape(()))), role="scalar", **self.update_meta(meta))
                if isinstance(data, np.ndarray) and data.ndim == 2 and 1 in data.shape and data.size > 1:
                    vec = np.ravel(data)
                    return VecOpInfo(vec, role=_value_role_from_vec(vec), **self.update_meta(meta))
                return GradOpInfo(data, role=self.role, **self.update_meta(meta))
            if other_grad_val.shape[0] == 1 and grad_val.shape[1] == other_grad_val.shape[1]:
                data = grad_val @ other_grad_val[0]
                return VecOpInfo(np.asarray(data), role=_value_role_from_vec(data), **self.update_meta(meta))
            if grad_val.shape[0] == 1 and grad_val.shape[1] == other_grad_val.shape[0]:
                data = grad_val[0] @ other_grad_val
                return VecOpInfo(np.asarray(data), role=_value_role_from_vec(data), **self.update_meta(meta))
            raise NotImplementedError(self._error_msg(other, "dot between gradients"))

        # Case 4: scalar basis-gradient · scalar basis-gradient -> mixed scalar block
        if self.role in {"trial", "test"} and other.role in {"trial", "test"} and _is_scalar_field(self) and _is_scalar_field(other):
            lhs_basis = _scalar_grad_basis_matrix(self)
            rhs_basis = _scalar_grad_basis_matrix(other)
            meta = _resolve_meta(self.meta(), other.meta())
            if self.role == "test" and other.role == "trial":
                data = lhs_basis.T @ rhs_basis
            elif self.role == "trial" and other.role == "test":
                data = rhs_basis.T @ lhs_basis
            else:
                raise NotImplementedError("GradOpInfo.dot between two trial or two test scalar gradients is not implemented.")
            return VecOpInfo(data, role="mixed", **self.update_meta(meta))

        # Case 5: vector/tensor basis-gradient · vector/tensor basis-gradient
        if self.role in {"trial", "test"} and other.role in {"trial", "test"} and self.shape[-1] == other.shape[0]:
            is_test = self.role == "test" or other.role == "test"
            is_trial = self.role == "trial" or other.role == "trial"
            if is_trial and is_test:
                role = "mixed"
                if self.role == "trial" and other.role == "test":
                    data = np.einsum("knd,dml->kmnl", self.data, other.data, optimize=True)
                else:
                    data = np.einsum("knd,dml->knml", self.data, other.data, optimize=True)
                return GradOpInfo(
                    data,
                    role=role,
                    field_names=self.field_names,
                    parent_name=self.parent_name,
                    side=self.side,
                    field_sides=self.field_sides,
                    is_rhs=self.is_rhs,
                )
            raise NotImplementedError("GradOpInfo.dot between two trial or two test functions is not implemented.")

        raise NotImplementedError(
            f"GradOpInfo.dot not implemented for roles {self.role} and {other.role}."
            f" Shapes: {self.data.shape} and {other.data.shape}."
        )

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
                if left_vec.role in {"trial", "test", "trial_n", "test_n"}:
                    meta = _resolve_meta(self.meta(), left_vec.meta(), prefer='b')
                    if grad_val.shape[0] == left_vec.shape[0]:
                        data = np.einsum("kn,kd->dn", left_vec.data, grad_val, optimize=True)
                        return VecOpInfo(data, role=left_vec.role, **left_vec.update_meta(meta))
                    # Scalar-gradient special case: grad(Function scalar) has shape (1,d),
                    # while vector trial/test has shape (d,n). Contract over d to get scalar basis (n).
                    if grad_val.shape[0] == 1 and left_vec.data.shape[0] == grad_val.shape[1]:
                        data = np.einsum("dn,d->n", left_vec.data, grad_val[0], optimize=True)
                        return VecOpInfo(data[np.newaxis, :], role=left_vec.role, **left_vec.update_meta(meta))
                    raise NotImplementedError(self._error_msg(left_vec, "left_dot with vector"))
                elif left_vec.role == "function":
                    u_vals = _collapsed_function(left_vec)  # shape (k,)
                    data = np.dot(u_vals, grad_val)  # (d,) result
                    role = "vector" if data.ndim == 1 else "scalar"
                    return VecOpInfo(data, role=role, **self.update_meta(meta))
            elif self.role in {"trial", "test", "trial_n", "test_n"}:
                if _is_scalar_field(self):
                    g_basis = _scalar_grad_basis_matrix(self)  # (d, n)
                    if left_vec.role in {"trial", "test", "trial_n", "test_n"}:
                        meta = _resolve_meta(self.meta(), left_vec.meta(), prefer='a')
                        if left_vec.role.startswith("trial") and self.role.startswith("test"):
                            data = np.einsum("dm,dn->nm", left_vec.data, g_basis, optimize=True)
                        elif left_vec.role.startswith("test") and self.role.startswith("trial"):
                            data = np.einsum("dn,dm->nm", left_vec.data, g_basis, optimize=True)
                        else:
                            data = np.einsum("dn,dm->nm", left_vec.data, g_basis, optimize=True)
                        return VecOpInfo(data, role="mixed", **self.update_meta(meta))
                    if left_vec.role == "function":
                        u_vals = _collapsed_function(left_vec)
                        if _is_1d_vector(u_vals) and u_vals.shape[0] == g_basis.shape[0]:
                            data = np.einsum("d,dn->n", u_vals, g_basis, optimize=True)
                            meta = _resolve_meta(self.meta(), left_vec.meta(), prefer='a')
                            return VecOpInfo(data[np.newaxis, :], role=self.role, **self.update_meta(meta))
                        raise NotImplementedError(self._error_msg(left_vec, "left_dot with vector"))
                if left_vec.role in {"trial", "test", "trial_n", "test_n"} and self.role in {"trial", "test", "trial_n", "test_n"}:
                    if self.data.shape[0] != left_vec.data.shape[0]:
                        raise NotImplementedError(self._error_msg(left_vec, "left_dot with vector"))
                    meta = _resolve_meta(self.meta(), left_vec.meta(), prefer='a')
                    if left_vec.role.startswith("test") and self.role.startswith("trial"):
                        data = np.einsum("km,knd->dmn", left_vec.data, self.data, optimize=True)
                    elif left_vec.role.startswith("trial") and self.role.startswith("test"):
                        data = np.einsum("km,knd->dnm", left_vec.data, self.data, optimize=True)
                    else:
                        data = np.einsum("km,knd->dmn", left_vec.data, self.data, optimize=True)
                    return VecOpInfo(data, role="mixed", **self.update_meta(meta))
                if left_vec.role == "function":
                    u_vals = _collapsed_function(left_vec)  # shape (k,)
                    if _is_1d_vector(u_vals):
                        if _is_scalar_field(self):
                            # Special case: single component gradient
                            data = np.einsum("d,kld->kl", u_vals, self.data, optimize=True)
                        elif self.data.shape[0] == u_vals.shape[0]:
                            data = np.einsum("s,snd->dn", u_vals, self.data, optimize=True)
                        else: raise NotImplementedError(self._error_msg(left_vec, "left_dot with vector"))
                        meta = _resolve_meta(self.meta(), left_vec.meta(), prefer='a')
                        return VecOpInfo(data, role=self.role, **left_vec.update_meta(meta))
                    else: raise NotImplementedError(self._error_msg(left_vec, "left_dot with vector"))
            elif self.role == "mixed":
                if left_vec.role in {"function", "vector"}:
                    vec_vals = _collapsed_function(left_vec) if left_vec.role == "function" else np.asarray(left_vec.data, dtype=float)
                    if vec_vals.ndim == 1 and vec_vals.shape[0] == self.data.shape[0]:
                        data = np.einsum("s,snmd->dnm", vec_vals, self.data, optimize=True)
                        meta = _resolve_meta(self.meta(), left_vec.meta(), prefer='a')
                        return VecOpInfo(data, role=self.role, **self.update_meta(meta))
                raise NotImplementedError(self._error_msg(left_vec, "left_dot with mixed tensor"))

        if isinstance(left_vec, np.ndarray) and left_vec.ndim == 1:
            if self.role == "function":
                if left_vec.shape[0] >1:
                    # Case:  Const · Grad(Function)      c · ∇u_k
                    grad_val = _collapsed_grad(self)  # shape (k, d)  —   ∇u_k(ξ)
                    if grad_val.shape[0] == 1:
                        # Special case: single component gradient
                        data = np.einsum("d,kd->k", left_vec, grad_val, optimize=True) # returns a 1D array
                        return VecOpInfo(data, role="vector", **self.update_meta(self.meta()))
                    else:
                        data = np.einsum("k,kd->d", left_vec, grad_val, optimize=True)
                    role = "vector" if data.ndim == 1 else "scalar"
                    return VecOpInfo(data, role=role, **self.update_meta(self.meta()))
                else:
                    raise ValueError(f"Cannot left_dot with vector of shape {left_vec.shape}."
                                     f" with roles {self.role} and {getattr(left_vec, 'role', None)}.")
            elif self.role in {"trial", "test"}:
                if left_vec.shape[0] >1:
                    if _is_scalar_field(self):
                        g_basis = _scalar_grad_basis_matrix(self)
                        data = np.einsum("d,dn->n", left_vec, g_basis, optimize=True)
                        meta = _resolve_meta(self.meta(), {}, prefer='a')
                        return VecOpInfo(data[np.newaxis, :], role=self.role, **self.update_meta(meta))
                    if self.data.shape[0] == 1:
                        # Special case: single component gradient
                        data = np.einsum("d,kld->kl", left_vec, self.data, optimize=True)
                    else:
                        data = np.einsum("s,snd->dn", left_vec, self.data, optimize=True)
                    meta = _resolve_meta(self.meta(), {}, prefer='a')
                    return VecOpInfo(data, role=self.role, **self.update_meta(meta))
                elif left_vec.shape[0] == 1:
                    # Special case: single component gradient
                    data = left_vec[0] * self.data  # broadcasting
                    return self._with(data, role=self.role)
                else:
                    raise ValueError(f"Cannot left_dot with vector of shape {left_vec.shape} and {self.shape}."
                                     f" with roles {self.role} and {getattr(left_vec, 'role', None)}.")
            elif self.role == "mixed":
                if left_vec.shape[0] != self.data.shape[0]:
                    raise ValueError(f"Cannot left_dot mixed GradOpInfo {self.shape} with vector of shape {left_vec.shape}.")
                data = np.einsum("s,snmd->dnm", left_vec, self.data, optimize=True)
                meta = _resolve_meta(self.meta(), {}, prefer='a')
                return VecOpInfo(data, role=self.role, **self.update_meta(meta))

        if isinstance(left_vec, np.ndarray) and left_vec.ndim == 2:
            if self.role in {"trial", "test"}:
                if left_vec.shape[1] != self.data.shape[0]:
                    raise ValueError(f"Cannot left_dot with matrix of shape {left_vec.shape}.")
                data = np.einsum("im,mnd->ind", left_vec, self.data, optimize=True)
                return self._with(data, role=self.role, coeffs=self.coeffs)
            if self.role == "function":
                grad_val = _collapsed_grad(self)
                if left_vec.shape[1] != grad_val.shape[0]:
                    raise ValueError(f"Cannot left_dot with matrix of shape {left_vec.shape}.")
                data = left_vec @ grad_val
                role = "vector" if data.ndim == 2 else "scalar"
                return VecOpInfo(data, role=role, **self.update_meta(self.meta()))
            if self.role == "mixed":
                if left_vec.shape[1] != self.data.shape[0]:
                    raise ValueError(f"Cannot left_dot mixed GradOpInfo {self.shape} with matrix of shape {left_vec.shape}.")
                data = np.einsum("im,mnrd->inrd", left_vec, self.data, optimize=True)
                return self._with(data, role=self.role, coeffs=self.coeffs)

        raise ValueError(f"Cannot left_dot with vector of shape {left_vec.shape}.")

    def dot_vec(self, other_vec: np.ndarray) -> VecOpInfo:
        """
        Computes the dot product with a constant vector or VecOpInfo over the SPATIAL dimension.
        This operation reduces the dimension and returns a VecOpInfo object.
        dot(∇v, c) -> ∇v ⋅ c
        """
        
        if isinstance(other_vec, (VecOpInfo)): # this part is until trial grad(u) dot u_k  ((∇u) · u_k)

            if self.role in {"function", "scalar"} and other_vec.role == "mixed":
                return other_vec.dot_grad(self)

            if self.role == "function" and other_vec.role == "trial": # introducing a new branch
                # Case:  Grad(Function) · Vec(Trial)      (∇u_k) · u
                grad_val = _collapsed_grad(self)  # shape (k, d) · (k,n)  —   ∇u_k(ξ)
                if grad_val.shape[-1] == other_vec.shape[0]:
                    data = grad_val @ other_vec.data
                    meta = _resolve_meta(self.meta(), other_vec.meta(), prefer='b')
                    return VecOpInfo(data, role=other_vec.role, **other_vec.update_meta(meta))
                else: raise NotImplementedError(self._error_msg(other_vec, "dot_vec"))
            elif self.role == "function" and other_vec.role == "test":
                # Case:  Grad(Function) · Vec(Test)       (∇u_k) · v
                grad_val = _collapsed_grad(self)
                if grad_val.shape[-1] == other_vec.shape[0]:
                    data = grad_val @ other_vec.data
                    meta = _resolve_meta(self.meta(), other_vec.meta(), prefer="b")
                    return VecOpInfo(data, role=other_vec.role, **other_vec.update_meta(meta))
                else:
                    raise NotImplementedError(self._error_msg(other_vec, "dot_vec"))
            
            elif self.role == "trial" and other_vec.role == "function": # introducing a new branch
                # Case:  Grad(Trial) · Vec(Function)      (∇u_trial) · u_k
                v_val = _collapsed_function(other_vec)  # shape (k,)  —   u_k(ξ)
                if _is_scalar_field(self):
                    g_basis = _scalar_grad_basis_matrix(self)
                    if g_basis.shape[0] != v_val.shape[0]:
                        raise NotImplementedError(self._error_msg(other_vec, "dot_vec"))
                    data = np.einsum("dn,d->n", g_basis, v_val, optimize=True)
                    meta = _resolve_meta(self.meta(), other_vec.meta(), prefer='a')
                    return VecOpInfo(data[np.newaxis, :], role=self.role, **other_vec.update_meta(meta))
                if self.data.shape[-1] != v_val.shape[0]:
                    raise NotImplementedError(self._error_msg(other_vec, "dot_vec"))
                data = np.einsum("kld,d->kl", self.data, v_val, optimize=True)
                meta = _resolve_meta(self.meta(), other_vec.meta(), prefer='a')
                return VecOpInfo(data, role=self.role, **other_vec.update_meta(meta))
            elif self.role == "test" and other_vec.role == "function":
                # Case:  Grad(Test) · Vec(Function)      (∇v_test) · u_k
                # Needed for skew-symmetric convection forms.
                v_val = _collapsed_function(other_vec)  # shape (k,)  —   u_k(ξ)
                if _is_scalar_field(self):
                    g_basis = _scalar_grad_basis_matrix(self)
                    if g_basis.shape[0] != v_val.shape[0]:
                        raise NotImplementedError(self._error_msg(other_vec, "dot_vec"))
                    data = np.einsum("dn,d->n", g_basis, v_val, optimize=True)
                    meta = _resolve_meta(self.meta(), other_vec.meta(), prefer="a")
                    return VecOpInfo(data[np.newaxis, :], role=self.role, **self.update_meta(meta))
                if self.data.shape[-1] != v_val.shape[0]:
                    raise NotImplementedError(self._error_msg(other_vec, "dot_vec"))
                data = np.einsum("knd,d->kn", self.data, v_val, optimize=True)
                meta = _resolve_meta(self.meta(), other_vec.meta(), prefer="a")
                return VecOpInfo(data, role=self.role, **self.update_meta(meta))
            elif self.role == "function" and other_vec.role == "function":
                # Case:  Grad(Function) · Vec(Function)      (∇u_k) · u_k
                # (1)  value of ∇u_k at this quad-point
                grad_val = _collapsed_grad(self)  # shape (k, d)  —   ∇u_k(ξ)
                v_val = _collapsed_function(other_vec)  # shape (k,)  —   u_k(ξ)
                if grad_val.shape[-1] != v_val.shape[0]:
                    raise NotImplementedError(self._error_msg(other_vec, "dot_vec"))
                data = np.einsum("kd,d->k", grad_val, v_val, optimize=True) # (k,) result
                if isinstance(data, np.ndarray) and data.shape == (1,):
                    return VecOpInfo(np.asarray(data[0]), role="scalar", **self.update_meta(self.meta()))
                role = "scalar" if data.ndim == 0 else "vector"
                return VecOpInfo(data, role=role, **self.update_meta(self.meta()))
            elif self.role in {"trial", "test"} and other_vec.role in {"trial", "test"}:
                if _is_scalar_field(self):
                    g_basis = _scalar_grad_basis_matrix(self)
                    if other_vec.role == "trial" and self.role == "test":
                        data = np.einsum("dm,dn->nm", other_vec.data, g_basis, optimize=True)
                    elif other_vec.role == "test" and self.role == "trial":
                        data = np.einsum("dn,dm->nm", other_vec.data, g_basis, optimize=True)
                    else:
                        data = np.einsum("dn,dm->nm", other_vec.data, g_basis, optimize=True)
                    meta = _resolve_meta(self.meta(), other_vec.meta(), prefer='a')
                    return VecOpInfo(data, role="mixed", **self.update_meta(meta))
                if self.data.shape[-1] != other_vec.data.shape[0]:
                    raise NotImplementedError(self._error_msg(other_vec, "dot_vec"))
                # Keep a consistent mixed orientation: rows=test, cols=trial.
                if self.role == "trial" and other_vec.role == "test":
                    data = np.einsum("knd,dm->kmn", self.data, other_vec.data, optimize=True)
                    meta = _resolve_meta(self.meta(), other_vec.meta(), prefer="b")
                elif self.role == "test" and other_vec.role == "trial":
                    data = np.einsum("knd,dm->knm", self.data, other_vec.data, optimize=True)
                    meta = _resolve_meta(self.meta(), other_vec.meta(), prefer="a")
                else:
                    data = np.einsum("knd,dm->knm", self.data, other_vec.data, optimize=True)
                    meta = _resolve_meta(self.meta(), other_vec.meta(), prefer='a')
                return VecOpInfo(data, role="mixed", **self.update_meta(meta))
            elif self.role in {"trial", "test"} and other_vec.role == "vector":
                vec_vals = np.asarray(other_vec.data)
                if _is_scalar_field(self):
                    g_basis = _scalar_grad_basis_matrix(self)
                    if vec_vals.ndim == 1 and vec_vals.shape[0] == g_basis.shape[0]:
                        contracted = np.einsum("dn,d->n", g_basis, vec_vals, optimize=True)
                        meta = _resolve_meta(self.meta(), other_vec.meta(), prefer='a')
                        return VecOpInfo(contracted[np.newaxis, :], role=self.role, **self.update_meta(meta))
                    raise NotImplementedError(self._error_msg(other_vec, "dot_vec with vector data"))
                if vec_vals.ndim == 1 and self.data.shape[0] == 1 and vec_vals.shape[0] == self.data.shape[-1]:
                    contracted = np.einsum("knd,d->kn", self.data, vec_vals, optimize=True)
                    meta = _resolve_meta(self.meta(), other_vec.meta(), prefer='a')
                    return VecOpInfo(contracted, role=self.role, **self.update_meta(meta))
                if vec_vals.ndim == 1 and vec_vals.shape[0] == self.data.shape[0]:
                    contracted = np.einsum("knd,d->kn", self.data, vec_vals, optimize=True)
                    meta = _resolve_meta(self.meta(), other_vec.meta(), prefer='a')
                    return VecOpInfo(contracted, role=self.role, **self.update_meta(meta))
                raise NotImplementedError(self._error_msg(other_vec, "dot_vec with vector data"))
            elif self.role == "function" and other_vec.role == "vector":
                grad_val = _collapsed_grad(self)
                vec_vals = np.asarray(other_vec.data)
                if vec_vals.ndim == 1 and grad_val.shape[-1] == vec_vals.shape[0]:
                    data = np.einsum("kd,d->k", grad_val, vec_vals, optimize=True)
                    if isinstance(data, np.ndarray) and data.shape == (1,):
                        return VecOpInfo(np.asarray(data[0]), role="scalar", **self.update_meta(self.meta()))
                    role = "scalar" if np.ndim(data) == 0 else "vector"
                    return VecOpInfo(data, role=role, **self.update_meta(self.meta()))
                raise NotImplementedError(self._error_msg(other_vec, "dot_vec with vector data"))
            else:
                raise NotImplementedError(
                    f"dot_vec of GradOpInfo not implemented for role {self.role} and VecOpInfo role {other_vec.role}."
                    f" Shapes: {self.data.shape} and {other_vec.data.shape}."
                )

            return self._with(self.data * other_vec.data, role=self.role)
        
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
                # Keep scalar outputs as arrays (0-D or 1-D) to preserve `.shape`
                # and avoid breaking VecOpInfo algebra in later sums/products.
                if isinstance(data, np.ndarray) and data.shape == (1,):
                    return VecOpInfo(np.asarray(data[0]), role="scalar", **self.update_meta(self.meta()))
                role = "scalar" if np.ndim(data) == 0 else "vector"
                return VecOpInfo(data, role=role, **self.update_meta(self.meta()))
            if self.role in {"trial", "test", "trial_n", "test_n"}:
                if _is_scalar_field(self):
                    g_basis = _scalar_grad_basis_matrix(self)
                    if g_basis.shape[0] != other_vec.shape[0]:
                        raise ValueError(
                            f"Cannot dot(∇w, n) scalar GradOpInfo {self.shape} with vector of shape {other_vec.shape}."
                        )
                    result_data = np.einsum("dn,d->n", g_basis, other_vec, optimize=True)
                    meta = _resolve_meta(self.meta(), {}, prefer='a')
                    return VecOpInfo(result_data[np.newaxis, :], role=self.role, **self.update_meta(meta))
                if self.data.shape[-1] != other_vec.shape[0]:
                    raise ValueError(f"Cannot dot(∇w, n) GradOpInfo {self.shape} with vector of shape {other_vec.shape}.")
                result_data = np.einsum("knd,d->kn", self.data, other_vec, optimize=True)
                meta = _resolve_meta(self.meta(), {}, prefer='a')
                return VecOpInfo(result_data, role=self.role, **self.update_meta(meta))
            if self.role == "mixed":
                result_data = np.einsum("knmd,d->knm", self.data, other_vec, optimize=True)
                meta = _resolve_meta(self.meta(), {}, prefer='a')
                return VecOpInfo(result_data, role=self.role, **self.update_meta(meta))
        if isinstance(other_vec, np.ndarray) and other_vec.ndim == 2:
            if self.role in {"trial", "test"}:
                if _is_scalar_field(self):
                    g_basis = _scalar_grad_basis_matrix(self)
                    if other_vec.shape[0] != g_basis.shape[0]:
                        raise NotImplementedError(self._error_msg(other_vec, "dot_vec"))
                    data = g_basis.T @ other_vec
                    return self._with(data[np.newaxis, ...], role=self.role, coeffs=self.coeffs)
                if other_vec.shape[0] != self.data.shape[-1]:
                    raise NotImplementedError(self._error_msg(other_vec, "dot_vec"))
                data = np.einsum("knd,dm->knm", self.data, other_vec, optimize=True)
                return self._with(data, role=self.role, coeffs=self.coeffs)
            if self.role == "function":
                grad_val = _collapsed_grad(self)
                if other_vec.shape[0] != grad_val.shape[-1]:
                    raise NotImplementedError(self._error_msg(other_vec, "dot_vec"))
                data = grad_val @ other_vec
                role = "vector" if data.ndim == 2 else "scalar"
                return VecOpInfo(data, role=role, **self.update_meta(self.meta()))
            if self.role == "mixed":
                if other_vec.shape[0] != self.data.shape[-1]:
                    raise NotImplementedError(self._error_msg(other_vec, "dot_vec"))
                return self._with(
                    np.einsum("...d,dm->...m", self.data, other_vec, optimize=True),
                    role=self.role,
                    coeffs=self.coeffs,
                )
        other_role = getattr(other_vec, "role", None)
        raise NotImplementedError(
            f"dot_vec of GradOpInfo not implemented for role {self.role}, type {type(other_vec)} with role: {other_role}."
            f" Shapes: {self.data.shape} and {getattr(other_vec, 'shape', None)}."
        )

    # ========================================================================
    # Dunder methods for SCALING Operations
    # ========================================================================

    def __mul__(self, other: Union[float, int, np.ndarray]) -> "GradOpInfo":
        """
        Performs SCALING and always returns a new GradOpInfo object.
        This operator preserves the shape (k, n, d).
        """
        if isinstance(other, (float, int, Constant)):
            # Case 1: Scaling by a single scalar
            new_data = self.data * other
            role = self.role
            
        elif isinstance(other, np.ndarray):
            other = np.asarray(other)
            role = self.role
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
            elif other.ndim == 2:
                if self.data.ndim >= 3:
                    if other.shape[0] != self.data.shape[-1]:
                        raise ValueError(
                            f"Cannot right-contract GradOpInfo(shape={self.shape}) with matrix of shape {other.shape}."
                        )
                    new_data = np.einsum("...d,dm->...m", self.data, other, optimize=True)
                elif self.data.ndim == 2:
                    if self.role in {"trial", "test"} and _is_scalar_field(self):
                        if other.shape[0] != self.data.shape[0]:
                            raise ValueError(
                                f"Cannot right-contract scalar GradOpInfo(shape={self.shape}) with matrix of shape {other.shape}."
                            )
                        new_data = other.T @ self.data
                    else:
                        if other.shape[0] != self.data.shape[-1]:
                            raise ValueError(
                                f"Cannot right-contract GradOpInfo(shape={self.shape}) with matrix of shape {other.shape}."
                            )
                        new_data = self.data @ other
                else:
                    raise ValueError(f"Cannot right-contract GradOpInfo(shape={self.shape}) with matrix of shape {other.shape}.")
            else:
                 raise ValueError(f"Cannot scale GradOpInfo(shape={self.shape}) with array of shape {other.shape}.")
        elif isinstance(other, VecOpInfo):
            if other.role == "scalar":
                # Scalar coefficient/value scaling a gradient table (e.g. grad(u)*c where
                # c is a collapsed coefficient function at a quadrature point).
                o = np.asarray(other.data)
                if o.ndim == 0:
                    scale = float(o)
                elif o.ndim == 1 and o.shape[0] == 1:
                    scale = float(o[0])
                else:
                    raise TypeError(
                        f"GradOpInfo can only be scaled by a scalar VecOpInfo; got shape {o.shape} for role {other.role}."
                    )
                return self._with(self.data * scale, role=self.role)
            if self.role in ("identity", "function") and other.role in ( "function"):
                # Case 4: Scaling by a vector of size d (e.g., a constant vector field)
                grad_vals = _collapsed_grad(self)  # shape (k, d)  —   ∇u_k(ξ)
                vec_vals = _collapsed_function(other)  # shape (k,)  —   u
                if vec_vals.shape[0] == 1:
                    # Scalar multiplication
                    new_data = grad_vals * vec_vals[0]
                    role = "function"
                else:
                    raise ValueError(f"Cannot scale GradOpInfo(shape={self.shape}) with VecOpInfo of shape {other.data.shape}.")
            elif self.role in ("identity", "function") and other.role in ( "trial", "test") and _is_scalar_field(other):
                # Case 5: Scale trial and test promote them to (k,n,d) shape
                grad_vals = _collapsed_grad(self)  # shape (k, d)  —
                new_data = grad_vals[:, np.newaxis, :] * other.data[:, :, np.newaxis]  # broadcasting
                role = other.role
            elif self.role in ("trial", "test") and other.role in ( "function"):
                # Case 6: Scale trial and test promote them to (k,n,d) shape
                vec_vals = _collapsed_function(other)  # shape (k,)  —   u
                if vec_vals.shape[0] == 1:
                    # Scalar multiplication
                    new_data = self.data * vec_vals[0]
                    role = self.role
                else:
                    raise ValueError(f"Cannot scale GradOpInfo(shape={self.shape}) with VecOpInfo of shape {other.data.shape}.")
            else:
                raise TypeError(f"GradOpInfo can only be scaled by a scalar or array, not VecOpInfo with role {other.role}.\n"
                                f"Got roles : a:{self.role} , b:{other.role}\n"
                                f"Shapes : a:{self.data.shape}, b: {other.data.shape}")

        else:
            raise TypeError(f"GradOpInfo can only be scaled by a scalar or array, not {type(other).__name__}.\n"
                            f"Got roles : a:{self.role} , b:{getattr(other, 'role', None), }\n"
                            f"Shapes : a:{self.data.shape}, b: {getattr(other, 'shape', None)}")

        return self._with(new_data, role=role)

    def __rmul__(self, other: Union[float, int, np.ndarray]) -> "GradOpInfo":
        if isinstance(other, np.ndarray):
            other = np.asarray(other)
            if other.ndim == 2:
                if self.data.ndim == 4:
                    if other.shape[1] != self.data.shape[0]:
                        raise ValueError(
                            f"Cannot left-contract matrix of shape {other.shape} with mixed/tensor GradOpInfo of shape {self.shape}."
                        )
                    new_data = np.einsum("mk,k...->m...", other, self.data, optimize=True)
                    return self._with(new_data, role=self.role)
                if self.data.ndim == 3:
                    if self.data.shape[0] == 1:
                        if other.shape[1] != self.data.shape[-1]:
                            raise ValueError(
                                f"Cannot left-contract matrix of shape {other.shape} with scalar gradient basis of shape {self.shape}."
                            )
                        new_data = np.einsum("md,...d->...m", other, self.data, optimize=True)
                        return self._with(new_data, role=self.role)
                    if other.shape[1] != self.data.shape[0]:
                        raise ValueError(
                            f"Cannot left-contract matrix of shape {other.shape} with vector/tensor gradient basis of shape {self.shape}."
                        )
                    new_data = np.einsum("mk,knd->mnd", other, self.data, optimize=True)
                    return self._with(new_data, role=self.role)
                if self.data.ndim == 2:
                    if self.role in {"trial", "test"} and _is_scalar_field(self):
                        if other.shape[1] != self.data.shape[0]:
                            raise ValueError(
                                f"Cannot left-contract matrix of shape {other.shape} with scalar gradient basis of shape {self.shape}."
                            )
                        new_data = other @ self.data
                        return self._with(new_data, role=self.role)
                    if self.data.shape[0] == 1:
                        if other.shape[1] != self.data.shape[-1]:
                            raise ValueError(
                                f"Cannot left-contract matrix of shape {other.shape} with scalar gradient value of shape {self.shape}."
                            )
                        new_data = self.data @ other.T
                        return self._with(new_data, role=self.role)
                    if other.shape[1] != self.data.shape[0]:
                        raise ValueError(
                            f"Cannot left-contract matrix of shape {other.shape} with vector/tensor gradient value of shape {self.shape}."
                        )
                    new_data = other @ self.data
                    return self._with(new_data, role=self.role)
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """Element-wise division of gradient tables by a scalar."""
        if isinstance(other, (float, int, np.floating)):
            return self._with(self.data / other, role=self.role, coeffs=self.coeffs)
        if isinstance(other, np.ndarray) and other.ndim == 0:
            return self._with(self.data / float(other), role=self.role, coeffs=self.coeffs)
        if isinstance(other, VecOpInfo) and other.role in {"function", "scalar"}:
            o = np.asarray(other.data)
            if o.ndim == 0:
                denom = float(o)
            elif o.ndim == 1 and o.shape[0] == 1:
                denom = float(o[0])
            else:
                raise TypeError(
                    f"GradOpInfo can only be divided by a scalar VecOpInfo; got shape {o.shape} for role {other.role}."
                )
            return self._with(self.data / denom, role=self.role, coeffs=self.coeffs)
        raise TypeError(f"GradOpInfo can only be divided by a scalar, not {type(other)}")

    def __rtruediv__(self, other):
        """Scalar divided by gradient tables (element-wise)."""
        if isinstance(other, (float, int, np.floating)):
            return self._with(other / self.data, role=self.role, coeffs=self.coeffs)
        if isinstance(other, np.ndarray) and other.ndim == 0:
            return self._with(float(other) / self.data, role=self.role, coeffs=self.coeffs)
        raise TypeError(f"Scalar division only; got numerator of type {type(other)}")
        
    def __neg__(self) -> "GradOpInfo":
        return self._with(-self.data, role=self.role)

    @staticmethod
    def _coerce_array_like(other):
        if isinstance(other, (int, float, np.floating)):
            return float(other)
        if isinstance(other, np.ndarray):
            return np.asarray(other, dtype=float)
        if isinstance(other, Constant):
            val = other.value
            if np.isscalar(val):
                return float(val)
            return np.asarray(val, dtype=float)
        if isinstance(other, Identity):
            return np.eye(other.size, dtype=float)
        return None

    def _add_array(self, array_like):
        if np.isscalar(array_like):
            scalar = float(array_like)
            if self.role in {"test", "trial"}:
                return self._with(self.data + scalar, role=self.role)
            if self.role == "function":
                mat = self._eval_function_to_2d()
                return self._with(mat + scalar, role=self.role)
            raise NotImplementedError(f"Scalar addition not implemented for role {self.role}.")

        arr = np.asarray(array_like, dtype=float)

        if self.role == "function":
            mat = self._eval_function_to_2d()
            if arr.shape == mat.shape:
                return self._with(mat + arr, role=self.role)
            if arr.ndim == 1 and mat.ndim == 2 and arr.shape[0] == mat.shape[1]:
                arr2 = np.broadcast_to(arr.reshape(1, -1), mat.shape)
                return self._with(mat + arr2, role=self.role)
            # if arr.ndim == 2 and arr.shape == (mat.shape[1], mat.shape[0]):
            #     arr = arr.T
            #     return self._with(mat + arr, role=self.role)
            raise ValueError(f"Cannot add array of shape {arr.shape} to function gradient of shape {mat.shape}.")

        if self.role in {"test", "trial"}:
            if arr.shape == self.data.shape:
                return self._with(self.data + arr, role=self.role)
            if arr.ndim == 2 and arr.shape == (self.data.shape[0], self.data.shape[2]):
                broadcast = arr[:, None, :]
                return self._with(self.data + broadcast, role=self.role)
            # if arr.ndim == 2 and arr.shape == (self.data.shape[2], self.data.shape[0]):
            #     broadcast = arr.T[:, None, :]
            #     return self._with(self.data + broadcast, role=self.role)
            # if arr.ndim == 1 and arr.shape[0] == self.data.shape[0]:
            #     broadcast = arr[:, None, None]
            #     return self._with(self.data + broadcast, role=self.role)
            raise ValueError(f"Cannot add array of shape {arr.shape} to gradient data of shape {self.data.shape}.")

        if self.role == "mixed":
            if arr.shape == self.data.shape:
                return self._with(self.data + arr, role=self.role)
            # if arr.ndim == 3 and arr.shape == (self.data.shape[0], self.data.shape[1], self.data.shape[3]):
            #     broadcast = arr[:, :, None, :]
            #     return self._with(self.data + broadcast, role=self.role)
            # if arr.ndim == 3 and arr.shape == (self.data.shape[0], self.data.shape[2], self.data.shape[3]):
            #     broadcast = arr[:, None, :, :]
            #     return self._with(self.data + broadcast, role=self.role)
            # if arr.ndim == 2 and arr.shape == (self.data.shape[0], self.data.shape[3]):
            #     broadcast = arr[:, None, None, :]
            #     return self._with(self.data + broadcast, role=self.role)
            # if arr.ndim == 2 and arr.shape == (self.data.shape[3], self.data.shape[0]):
            #     broadcast = arr.T[:, None, None, :]
            #     return self._with(self.data + broadcast, role=self.role)
            raise ValueError(f"Cannot add array of shape {arr.shape} to mixed gradient data of shape {self.data.shape}.")

        raise NotImplementedError(f"Array addition not implemented for role {self.role}.")

    def __add__(self, other) -> "GradOpInfo":
        if isinstance(other, BaseOpInfo):
            try:
                from pycutfem.ufl.tensor_algebra import TensorRuleEngine

                sum_lowering = TensorRuleEngine.plan_sum_lowering(self, other)
            except Exception:
                sum_lowering = None
            if sum_lowering is not None:
                lhs = _apply_storage_transform_opinfo(self, sum_lowering.lhs_transform)
                rhs = _apply_storage_transform_opinfo(other, sum_lowering.rhs_transform)
                if lhs is not self or rhs is not other:
                    return lhs.__add__(rhs)
        if isinstance(other, GradOpInfo):
            if self.role != other.role:
                if self.data.ndim < other.data.ndim:
                    return other.__add__(self)

                if other.data.shape == self.data.shape:
                    priority = ["mixed", "trial", "test", "function"]
                    for candidate in priority:
                        if candidate in {self.role, other.role}:
                            role = candidate
                            break
                    else:
                        raise ValueError(f"Addition of {self.role} and {other.role} GradOpInfo not supported.")
                    return self._with(self.data + other.data, role=role)

                if other.data.ndim <= 2:
                    return self._add_array(other.data)

                raise ValueError("Operands must be GradOpInfo of the same role for addition "
                                 f"(got {self.role} and {other.role}).")
            if self.role in {"test", "trial", "mixed"}:
                if self.data.shape != other.data.shape:
                    raise ValueError(f"GradOpInfo shapes mismatch in addition: {self.data.shape} vs {other.data.shape}.")
                return self._with(self.data + other.data, role=self.role)
            if self.role == "function":
                if other.role != "function":
                    raise ValueError(f"Cannot add GradOpInfo of role 'function' with role '{other.role}'.")
                a = self._eval_function_to_2d()
                b = other._eval_function_to_2d()
                if a.shape != b.shape:
                    raise ValueError(f"Function gradient shapes mismatch in addition: {a.shape} vs {b.shape}.")
                return self._with(a + b, role=self.role)
            raise NotImplementedError(f"GradOpInfo addition not implemented for role {self.role}.")

        # arr = self._coerce_array_like(other)
        # if arr is not None:
        #     return self._add_array(arr)
        arr = self._coerce_array_like(other)
        if arr is not None:
            return self._add_array(arr)

        raise TypeError(f"Unsupported type for addition with GradOpInfo: {type(other)}"
                        f" (role={self.role}), with shape {self.data.shape}."
                        f" other_role={getattr(other, 'role', None)}, other_shape={getattr(other, 'shape', None)}")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other: "GradOpInfo") -> "GradOpInfo":
        if other is None:
            raise TypeError("Attempting to subtract None from GradOpInfo — upstream visitor returned None")
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

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





@dataclass(slots=True, frozen=True)
class HessOpInfo(BaseOpInfo):
    """
    Hessian of basis/functions per component.
    data:
      - (k, n, d, d) for test/trial (per-dof Hessian tables)
      - (k, n, d, d) or (k, d, d)  for function (tables; coeffs may collapse n upstream)
    role: "test", "trial", or "function"
    coeffs: optional (k, n) — kept for parity with GradOpInfo (not applied here)
    """
    coeffs: np.ndarray = field(default=None)

    # ---------- helpers ----------
    def _with(self, data: np.ndarray, role: str | None = None, coeffs=None) -> "HessOpInfo":
        return HessOpInfo(
            np.asarray(data), role=(role or self.role),
            coeffs=(self.coeffs if coeffs is None else coeffs),
            field_names=self.field_names, parent_name=self.parent_name,
            side=self.side, field_sides=self.field_sides, is_rhs=self.is_rhs
        )

    @staticmethod
    def _as_dir_vec(vec) -> np.ndarray:
        """Return a 1D spatial vector (d,) from ndarray or VecOpInfo."""
        # ndarray → flatten
        if isinstance(vec, np.ndarray):
            v = np.asarray(vec).reshape(-1)
            if v.ndim != 1:
                raise ValueError(f"direction vector must be 1D, got {vec.shape}")
            return v
        # VecOpInfo → accept 'vector' or 'function'
        if isinstance(vec, VecOpInfo):
            if vec.role == "vector":
                return np.asarray(vec.data).reshape(-1)
            if vec.role == "function":
                # collapse components; require k == d to serve as direction
                w = _collapsed_function(vec)  # (k,)
                return np.asarray(w).reshape(-1)
        raise TypeError(f"Unsupported direction-vector type/role: {type(vec).__name__} (role={getattr(vec,'role',None)})")

    # ---------- linear algebra ops on the Hessian tensor -----------------

    def transpose(self) -> "HessOpInfo":
        """Swap the last two (spatial) axes i↔j; keep (k, n) intact."""
        return self._with(self.data.swapaxes(-1, -2), role=self.role, coeffs=self.coeffs)

    def __neg__(self) -> "HessOpInfo":
        return self._with(-self.data, role=self.role, coeffs=self.coeffs)

    def __mul__(self, other):
        """Scalar multiplication (left or right)."""
        if isinstance(other, (int, float, np.floating)):
            return self._with(self.data * other, role=self.role, coeffs=self.coeffs)
        if isinstance(other, np.ndarray) and other.ndim == 0:
            return self._with(self.data * float(other), role=self.role, coeffs=self.coeffs)
        raise TypeError(f"HessOpInfo can only be multiplied by scalars, not {type(other)}")
    __rmul__ = __mul__

    def __truediv__(self, other):
        """Element-wise division of Hessian tables by a scalar."""
        if isinstance(other, (int, float, np.floating)):
            return self._with(self.data / other, role=self.role, coeffs=self.coeffs)
        if isinstance(other, np.ndarray) and other.ndim == 0:
            return self._with(self.data / float(other), role=self.role, coeffs=self.coeffs)
        raise TypeError(f"HessOpInfo can only be divided by a scalar, not {type(other)}")

    def __rtruediv__(self, other):
        """Scalar divided by Hessian tables (element-wise)."""
        if isinstance(other, (int, float, np.floating)):
            return self._with(other / self.data, role=self.role, coeffs=self.coeffs)
        if isinstance(other, np.ndarray) and other.ndim == 0:
            return self._with(float(other) / self.data, role=self.role, coeffs=self.coeffs)
        raise TypeError(f"Scalar division only; got numerator of type {type(other)}")


    def __add__(self, other: "HessOpInfo") -> "HessOpInfo":
        if not isinstance(other, HessOpInfo):
            raise TypeError(f"Cannot add HessOpInfo to {type(other)}.")
        if self.data.shape != other.data.shape:
            raise ValueError(f"HessOpInfo shapes mismatch in addition: {self.data.shape} vs {other.data.shape}.")
        if self.role != other.role:
            raise ValueError(f"Cannot add HessOpInfo with different roles: {self.role} vs {other.role}.")
        # merge metadata; keep coeffs only if identical object
        coeffs = self.coeffs if (self.coeffs is other.coeffs) else None
        meta = _resolve_meta(self.meta(), other.meta(), strict=False)
        return HessOpInfo(self.data + other.data, role=self.role, coeffs=coeffs, **self.update_meta(meta))

    def __sub__(self, other: "HessOpInfo") -> "HessOpInfo":
        return self.__add__(other.__neg__())

    # ---------- algebra used in assembly ---------------------------------
    def trace(self) -> "VecOpInfo":
        """
        Trace over spatial axes → per-component Laplacian table:
          test/trial: (k, n)
          function:   (k, n) or (k,) depending on upstream collapsing
        Note: we *do not* apply coeffs here; contraction is handled by the caller.
        """
        tr = self.data[..., 0, 0] + self.data[..., 1, 1]
        # carry self's metadata forward
        return VecOpInfo(np.asarray(tr), role=self.role, **self.update_meta(self.meta()))

    # ---------- vector contractions --------------------------------------
    def dot_right(self, vec) -> "GradOpInfo":
        """
        Right contraction H · n over the last spatial axis j.
        Returns:
          test/trial → GradOpInfo with data (k,n,d)
          function   → GradOpInfo with data (k,d) if (k,d,d) else (k,n,d)
        """
        n = self._as_dir_vec(vec)  # (d,)
        if self.data.ndim == 4:    # (k,n,d,d) · (d,) -> (k,n,d)
            out = np.einsum("knij,j->kni", self.data, n, optimize=True)
            coeffs = self.coeffs
        elif self.data.ndim == 3:  # (k,d,d)   · (d,) -> (k,d)
            out = np.einsum("kij,j->ki",   self.data, n, optimize=True)
            coeffs = None
        else:
            raise ValueError(f"HessOpInfo.dot_right: unexpected ndim={self.data.ndim}")
        meta = _resolve_meta(self.meta(), getattr(vec, "meta", lambda: {})(), prefer='a')
        return GradOpInfo(np.asarray(out), role=self.role, coeffs=coeffs, **self.update_meta(meta))

    def dot_left(self, vec) -> "GradOpInfo":
        """
        Left contraction n · H over the first spatial axis i.
        Returns:
          test/trial → GradOpInfo with data (k,n,d)
          function   → GradOpInfo with data (k,d) if (k,d,d) else (k,n,d)
        """
        n = self._as_dir_vec(vec)  # (d,)
        is_vec = _is_1d_vector(n)
        is_scalar_field = _is_scalar_field(self)
        is_d_k = n.shape[0] == self.data.shape[0]
        is_d_d = n.shape[0] == self.data.shape[1] if self.data.ndim == 3 else n.shape[0] == self.data.shape[2]
        # is_d_dd = n.shape[0] == self.data.shape[2] if self.data.ndim == 3 else n.shape[0] == self.data.shape[3]

        if self.data.ndim == 4:    # (d,) · (k,n,d,d) -> (k,n,d)
            if is_vec and is_scalar_field and is_d_d:
                out = np.einsum("s,knsj->knj", n, self.data, optimize=True)
            elif is_vec and is_d_k:
                out = np.einsum("s,snij->inj", n, self.data, optimize=True)
            else: raise NotImplementedError(f"HessOpInfo.dot_left: cannot contract with vector of shape {n.shape} and Hess shape {self.data.shape}.")
            coeffs = self.coeffs
        elif self.data.ndim == 3:  # (d,) · (k,d,d)   -> (k,d)
            if is_vec and is_scalar_field and is_d_d:
                out = np.einsum("s,ksj->kj",   n, self.data, optimize=True)
            elif is_vec and is_d_k:
                out = np.einsum("s,sij->ij", n, self.data, optimize=True)
            else:
                raise NotImplementedError(f"HessOpInfo.dot_left: cannot contract with vector of shape {n.shape} and Hess shape {self.data.shape}.")
            coeffs = None
        else:
            raise ValueError(f"HessOpInfo.dot_left: unexpected ndim={self.data.ndim}")
        meta = _resolve_meta(self.meta(), getattr(vec, "meta", lambda: {})(), prefer='a')
        return GradOpInfo(np.asarray(out), role=self.role, coeffs=coeffs, **self.update_meta(meta))

    def proj_nn(self, nvec) -> "VecOpInfo":
        """
        Double contraction nᵀ H n (second normal derivative).
        Returns:
          test/trial → VecOpInfo with data (k,n)
          function   → VecOpInfo with data (k,) if (k,d,d) else (k,n)
        """
        n = self._as_dir_vec(nvec)  # (d,)
        if self.data.ndim == 4:    # (d,) · (k,n,d,d) · (d,) -> (k,n)
            tmp = np.einsum("i,knij->knj", n, self.data, optimize=True)
            val = np.einsum("knj,j->kn",   tmp, n, optimize=True)
        elif self.data.ndim == 3:  # (d,) · (k,d,d)   · (d,) -> (k,)
            tmp = np.einsum("i,kij->kj",   n, self.data, optimize=True)
            val = np.einsum("kj,j->k",     tmp, n, optimize=True)
        else:
            raise ValueError(f"HessOpInfo.proj_nn: unexpected ndim={self.data.ndim}")
        meta = _resolve_meta(self.meta(), getattr(nvec, "meta", lambda: {})(), prefer='a')
        return VecOpInfo(np.asarray(val), role=self.role, **self.update_meta(meta))

    def inner(self, other: "HessOpInfo") -> np.ndarray:
        """
        Frobenius inner product per DOF pair with proper orientation.
        Returns (n_test, n_trial), regardless of operand order.
        """
        if not isinstance(other, HessOpInfo):
            raise TypeError(f"Cannot take HessOpInfo.inner with {type(other)}.")
        if self.is_rhs:
            if self.role in {"function"} and other.role in { "trial", "test"}:
                kdij = _collapsed_hess(self)  # (k,d,d)
                if kdij.shape[0] != other.shape[0]:
                    raise NotImplementedError(self._error_msg(other, "inner with function Hessian"))
                A, B = kdij, other.data
                return np.einsum("kij,knij->n", A, B, optimize=True)
            elif self.role in {"trial", "test"} and other.role in {"function"}:
                kdij = _collapsed_hess(other)  # (k,d,d)
                if kdij.shape[0] != self.shape[0]:
                    raise NotImplementedError(self._error_msg(other, "inner with function Hessian"))
                A, B = self.data, kdij
                return np.einsum("knij,kij->n", A, B, optimize=True)
            elif self.role in {"function"} and other.role in {"function"}:
                kdij = _collapsed_hess(self)  # (k,d,d)
                mdij = _collapsed_hess(other)  # (k,d,d)
                if kdij.shape[0] != mdij.shape[0]:
                    raise NotImplementedError(self._error_msg(other, "inner with function Hessian"))
                A, B = kdij, mdij
                return np.einsum("kij,kij->", A, B, optimize=True)
            else: raise NotImplementedError(self._error_msg(other, "inner with function Hessian"))

        A, B = self.data, other.data
        if A.ndim != 4 or B.ndim != 4:
            raise ValueError("HessOpInfo.inner expects (k,n,d,d) arrays.")
        if A.shape[0] != B.shape[0] or A.shape[2:] != B.shape[2:]:
            raise ValueError(f"HessOpInfo component/spatial mismatch: {A.shape} vs {B.shape}")

        # Rows = test, Cols = trial
        if self.role == "test" and other.role == "trial":
            return np.einsum("knij,kmij->nm", A, B, optimize=True)
        if self.role == "trial" and other.role == "test":
            return np.einsum("knij,kmij->mn", A, B, optimize=True)
        if self.role == "function" and other.role == "function":
            # Rare path: collapse over (i,j) to component inner
            return np.einsum("kij,mij->km", A.sum(axis=1) if A.ndim==4 else A,
                                         B.sum(axis=1) if B.ndim==4 else B, optimize=True)
        raise NotImplementedError(
            f"HessOpInfo.inner not implemented for roles {self.role} and {other.role}."
        )

    # ---------- misc ----------
    @property
    def shape(self): return self.data.shape
    @property
    def ndim(self): return self.data.ndim

    def __repr__(self):
        return (f"HessOpInfo(shape={self.data.shape}, role='{self.role}')"
                f" (parent: {self.parent_name}, side: {self.side})"
                f" (fields: {self.field_names}, sides: {self.field_sides})"
                f" (is_rhs: {self.is_rhs})")

    def info(self):
        return (f"HessOpInfo({self.data.dtype}, shape={self.data.shape}, role='{self.role}')"
                f" (parent: {self.parent_name}, side: {self.side})"
                f" (fields: {self.field_names}, sides: {self.field_sides})"
                f" (is_rhs: {self.is_rhs})")



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
        
        if isinstance(node, UFLGrad):
            out.update({(1, 0), (0, 1)})
            _walk(node.operand, acc_x, acc_y)
            return
        if isinstance(node, UFLDiv):
            # Divergence is implemented via first derivatives of the operand.
            out.update({(1, 0), (0, 1)})
            _walk(node.operand, acc_x, acc_y)
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
    from pycutfem.ufl.expressions import HdivTrialFunction, HdivTestFunction
    trial = expr.find_first(lambda n: isinstance(n, (TrialFunction, VectorTrialFunction, HdivTrialFunction))) 
    test = expr.find_first(lambda n: isinstance(n, (TestFunction, VectorTestFunction, HdivTestFunction))) 
    return trial, test

# ------------------------------------------------------------------
#  find_all_restrictions -------------------------------------------
# ------------------------------------------------------------------
def _find_all_restrictions(form) -> list[Restriction]:
    restrictions = []
    visited = set()
    stack = [form]

    while stack:
        obj = stack.pop()
        oid = id(obj)
        if oid in visited:
            continue
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

        stack.extend(iterable)

    return restrictions


def analyze_active_dofs(
    equation: Equation,
    dh: DofHandler,
    me: MixedElement,
    bcs: list,
    *,
    verbose: bool = True,
):
    """Return (active_dofs, has_restriction) for reduced-system assembly.

    When no ``Restriction`` nodes are present we simply mark every DOF as
    active.  Otherwise, we examine each integral separately and keep **all**
    fields that appear in that integrand, evaluated on the union of the
    element domains referenced by its Restrictions.  This ensures auxiliary
    fields (e.g. global Lagrange multipliers) that couple to restricted
    fields remain active in the reduced system.
    """

    # Collect integrals from both sides of the equation
    integrals = list(getattr(equation.a, "integrals", [])) + \
                list(getattr(equation.L, "integrals", []))

    active_dofs: Set[int] = set()
    saw_restriction = False

    # XFEM: restrictions must account for enriched DOFs on cut elements.
    xfem_active = False
    xfem_me = None
    try:
        if hasattr(dh, "n_enriched") and callable(getattr(dh, "n_enriched")) and dh.n_enriched() > 0:
            if hasattr(dh, "xfem_mixed_element") and callable(getattr(dh, "xfem_mixed_element")):
                xfem_me = dh.xfem_mixed_element()
                xfem_active = xfem_me is not None
    except Exception:
        xfem_active = False
        xfem_me = None
    mesh = getattr(getattr(dh, "mixed_element", None), "mesh", None)

    for integral in integrals:
        rnodes = _find_all_restrictions(integral.integrand)
        if not rnodes:
            continue

        saw_restriction = True

        # Union of element ids touched by any Restriction in this integrand
        elem_ids: Set[int] = set()
        for r in rnodes:
            elem_ids.update(np.asarray(r.domain.to_indices(), dtype=int).tolist())

        if not elem_ids:
            continue

        fields = _all_fields(integral.integrand)

        for eid in elem_ids:
            ee = int(eid)
            element_dofs = None
            me_use = me
            if xfem_active and xfem_me is not None and mesh is not None:
                try:
                    tag = str(getattr(mesh.elements_list[ee], "tag", ""))
                except Exception:
                    tag = ""
                if tag == "cut" and hasattr(dh, "get_elemental_dofs_xfem"):
                    try:
                        element_dofs = dh.get_elemental_dofs_xfem(ee)
                        me_use = xfem_me
                    except Exception:
                        element_dofs = None
                        me_use = me
            if element_dofs is None:
                element_dofs = dh.get_elemental_dofs(ee)
            for field in fields:
                sl = me_use.component_dof_slices.get(field)
                if sl is None:
                    continue
                active_dofs.update(element_dofs[sl])

    if not saw_restriction:
        if verbose:
            print("No Restriction operators found. All DOFs are considered active.")
        return np.arange(dh.total_dofs, dtype=int), False

    if verbose:
        print("Restriction operators found. Analyzing active domains...")
    if not active_dofs:
        # Fall back to everything if analysis produced nothing (shouldn't happen).
        return np.arange(dh.total_dofs, dtype=int), True

    return np.array(sorted(active_dofs), dtype=int), True

from pycutfem.fem.transform import JET_CACHE, RefDerivCache
from pycutfem.fem import transform



def phys_scalar_third_row(me, fld: str, xi: float, eta: float,
                          i: int, j: int, k: int, mesh, elem_id: int,
                          ref_cache: RefDerivCache | None = None,
                          geom_jets: dict | None = None) -> np.ndarray:
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

    rec = geom_jets if geom_jets is not None else JET_CACHE.get(mesh, elem_id, xi, eta, upto=3)
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
                           ref_cache: RefDerivCache | None = None,
                           geom_jets: dict | None = None) -> np.ndarray:
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

    rec = geom_jets if geom_jets is not None else JET_CACHE.get(mesh, elem_id, xi, eta, upto=4)
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
    @lru_cache(maxsize=128)
    def _field_ref_nodes(element_type: str, p: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Reference-node coordinates for scalar Lagrange nodes in the ordering used
        by `pycutfem.fem.reference` (quads: eta outer, xi inner; tris: j outer, i inner).
        """
        import numpy as np

        element_type = str(element_type)
        p = int(p)
        if p < 0:
            raise ValueError("p must be non-negative")
        if element_type == "quad":
            if p == 0:
                xi = np.array([0.0], dtype=float)
                eta = np.array([0.0], dtype=float)
                return xi, eta
            nodes = np.linspace(-1.0, 1.0, p + 1, dtype=float)
            xi_list = []
            eta_list = []
            for et in nodes:
                for xs in nodes:
                    xi_list.append(float(xs))
                    eta_list.append(float(et))
            return np.asarray(xi_list, dtype=float), np.asarray(eta_list, dtype=float)
        if element_type == "tri":
            if p == 0:
                xi = np.array([0.0], dtype=float)
                eta = np.array([0.0], dtype=float)
                return xi, eta
            xi_list = []
            eta_list = []
            for j in range(p + 1):
                for i in range(p + 1 - j):
                    xi_list.append(float(i) / float(p))
                    eta_list.append(float(j) / float(p))
            return np.asarray(xi_list, dtype=float), np.asarray(eta_list, dtype=float)
        raise KeyError(f"Unsupported element_type '{element_type}'")

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
        tol_eff = SIDE.tol if tol is None else float(tol)
        ls_tol = getattr(level_set, "edge_tol", None)
        if ls_tol is None:
            ls_tol = getattr(level_set, "tol", None)
        if ls_tol is not None:
            try:
                tol_eff = max(tol_eff, float(ls_tol))
            except Exception:
                pass
        pos_masks: Dict[str, np.ndarray] = {}
        neg_masks: Dict[str, np.ndarray] = {}

        # Fast path: FE-backed level set with element-aware evaluation.
        mesh = getattr(getattr(dh, "mixed_element", None), "mesh", None)
        have_val_elem = hasattr(level_set, "value_on_element")
        have_vals_many = hasattr(level_set, "values_on_element_many")
        fast_eval = have_val_elem and (mesh is not None) and (getattr(mesh, "element_type", None) in {"quad", "tri"})
        families = getattr(getattr(dh, "mixed_element", None), "_field_families", {}) or {}

        for fld in fields:
            try:
                gidx = np.asarray(dh.element_maps[fld][eid], dtype=int)
            except Exception:
                continue

            # H(div) RT DOFs are facet/cell moments (not point evaluations). There is no
            # meaningful "DOF coordinate" to classify against φ, and RT is not currently
            # split/enriched by side masks. Treat all RT DOFs as active on both sides.
            try:
                if isinstance(families, dict) and families.get(str(fld)) == "RT":
                    pos_masks[fld] = np.ones((gidx.size,), dtype=float)
                    neg_masks[fld] = np.ones((gidx.size,), dtype=float)
                    continue
            except Exception:
                pass

            if fast_eval:
                p_f = int(getattr(dh.mixed_element, "_field_orders", {}).get(fld, 0))
                xi_ref, eta_ref = HelpersFieldAware._field_ref_nodes(str(mesh.element_type), p_f)
                if xi_ref.shape[0] != gidx.shape[0]:
                    # Fallback to the slow geometric path if orders/layouts mismatch.
                    fast_eval_this = False
                else:
                    fast_eval_this = True

                if fast_eval_this:
                    if have_vals_many:
                        phi = np.asarray(level_set.values_on_element_many(int(eid), xi_ref, eta_ref), dtype=float)
                    else:
                        phi = np.asarray(
                            [float(level_set.value_on_element(int(eid), (float(xi), float(eta))))
                             for (xi, eta) in zip(xi_ref, eta_ref)],
                            dtype=float,
                        )
                else:
                    phi = None
            else:
                phi = None

            if phi is None:
                # Slow fallback: evaluate at physical DOF coordinates (generic LS / exotic layouts).
                coords = dh.get_all_dof_coords()
                xy = coords[gidx]
                from pycutfem.ufl.helpers_geom import phi_eval
                phi = np.asarray([phi_eval(level_set, p) for p in xy], dtype=float)

            # Apply the convention (bulk sign) and include interface DOFs on both sides.
            if SIDE.pos_is_phi_nonnegative:
                pos_mask = (phi > 0.0).astype(float)
                neg_mask = (phi < 0.0).astype(float)
            else:
                pos_mask = (phi < 0.0).astype(float)
                neg_mask = (phi > 0.0).astype(float)
            interface_idx = np.abs(phi) <= float(tol_eff)
            if np.any(interface_idx):
                pos_mask[interface_idx] = 1.0
                neg_mask[interface_idx] = 1.0
            pos_masks[fld] = pos_mask
            neg_masks[fld] = neg_mask

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
    def _filter_fields_for_side(self: "FormCompiler", n, fields):
        """
        New rule: never guess from names. If a side is active in the context,
        we keep components and let _basis_row apply the correct side via
        (pos/neg) eid + per-field maps/masks. This prevents empty (0, n)
        blocks in Pos/Neg branches of Jump on ghost edges.
        """
        side_ctx = getattr(self, "ctx", {}).get("side", None)
        if side_ctx in ("+", "-"):
            return fields  # do not drop rows – padding/masks handle the side

        # If the node explicitly declares a single side for *all* components,
        # it’s still safe to keep them (evaluation will respect eid/side).
        side_node = getattr(n, "side", None)
        if side_node in ("+", "-"):
            return fields

        # Default: no filtering
        return fields

class HelpersAlignCoefficents:
    @staticmethod
    def _target_len_from_ctx(ctx):
        gd = ctx.get("global_dofs", None)
        return len(gd) if gd is not None else None
    @staticmethod
    def _field_map(ctx, side, field_name):
        # Per-field map: local(field) -> global
        return (ctx.get("pos_map_by_field", {}) if side == '+' else ctx.get("neg_map_by_field", {})).get(field_name, None)

    @staticmethod
    def _side_map(ctx, side):
        # Side-wide map: element-union(side) -> global
        return ctx.get("pos_map") if side == '+' else ctx.get("neg_map")

    @staticmethod
    def _pad_basis_to_global(ctx, side, field_name, phi):
        """
        Ensure basis 'phi' for 'field_name' is padded to the global layout when on
        ghost/interface paths (i.e., ctx has 'global_dofs').

        Supports 1D (ndofs,) and 2D (ncomp, ndofs) arrays.
        """
        tgt = HelpersAlignCoefficents._target_len_from_ctx(ctx)
        if tgt is None:
            return phi  # nothing to do

        fmap_full = HelpersAlignCoefficents._field_map(ctx, side, field_name)

        # Already global-sized? apply field mask to ensure other slots are zeroed.
        if phi.ndim == 1 and phi.shape[0] == tgt:
            if fmap_full is not None:
                mask = np.zeros(tgt, dtype=phi.dtype)
                mask[np.asarray(fmap_full, dtype=int)] = 1.0
                return phi * mask
            return phi
        if phi.ndim == 2 and phi.shape[1] == tgt:
            if fmap_full is not None:
                mask = np.zeros(tgt, dtype=phi.dtype)
                mask[np.asarray(fmap_full, dtype=int)] = 1.0
                return phi * mask[np.newaxis, :]
            return phi

        # If the vector is still in "side element" ordering, promote it using the side map
        if phi.ndim == 1 and side in ('+', '-'):
            fmap_first = HelpersAlignCoefficents._field_map(ctx, side, field_name)
            if fmap_first is not None and len(fmap_first) == phi.shape[0]:
                full = np.zeros(tgt, dtype=phi.dtype)
                full[np.asarray(fmap_first, dtype=int)] = phi
                phi = full
                return phi
            side_map = ctx.get('pos_map') if side == '+' else ctx.get('neg_map')
            if side_map is not None and len(side_map) == phi.shape[0]:
                full = np.zeros(tgt, dtype=phi.dtype)
                full[np.asarray(side_map, dtype=int)] = phi
                phi = full
                if phi.shape[0] == tgt:
                    return phi
        
        fmap = fmap_full
        if fmap is None:
            raise ValueError(f"Missing per-field map for basis padding: field='{field_name}', side={side}")

        fmap = np.asarray(fmap, dtype=np.intp)

        if phi.ndim == 1:
            if len(fmap) != phi.shape[0]:
                raise ValueError(f"Basis length mismatch for field '{field_name}': len(map)={len(fmap)} vs len(phi)={phi.shape[0]}")
            padded = np.zeros(tgt, dtype=phi.dtype)
            padded[fmap] = phi
            return padded

        elif phi.ndim == 2:
            if len(fmap) != phi.shape[1]:
                raise ValueError(f"Basis (2D) width mismatch for field '{field_name}': len(map)={len(fmap)} vs phi.shape[1]={phi.shape[1]}")
            padded = np.zeros((phi.shape[0], tgt), dtype=phi.dtype)
            padded[:, fmap] = phi
            return padded

        else:
            raise ValueError(f"Unsupported basis ndim={phi.ndim} for field '{field_name}'")


    @staticmethod
    def _pad_coeffs_to_global(ctx, side, field_name, u_loc):
        """
        Ensure the coefficient vector(s) 'u_loc' are padded into the global layout.

        Prefers the side-wide map (pos/neg_map) since u_loc is element-union sized.
        Falls back to per-field map only when lengths match.

        Supports 1D (ndofs,) and 2D (ncomp, ndofs).
        """
        tgt = HelpersAlignCoefficents._target_len_from_ctx(ctx)
        if tgt is None:
            return u_loc  # nothing to do

        fmap_full = HelpersAlignCoefficents._field_map(ctx, side, field_name)

        # Already global-sized? mask out slots outside the field.
        if u_loc.ndim == 1 and u_loc.shape[0] == tgt:
            if fmap_full is not None:
                mask = np.zeros(tgt, dtype=u_loc.dtype)
                mask[np.asarray(fmap_full, dtype=int)] = 1.0
                return u_loc * mask
            return u_loc
        if u_loc.ndim == 2 and u_loc.shape[1] == tgt:
            if fmap_full is not None:
                mask = np.zeros(tgt, dtype=u_loc.dtype)
                mask[np.asarray(fmap_full, dtype=int)] = 1.0
                return u_loc * mask[np.newaxis, :]
            return u_loc

        # Choose an indexer whose length matches u_loc's width
        smap = HelpersAlignCoefficents._side_map(ctx, side)
        fmap = fmap_full

        # Normalize to numpy int arrays if present
        smap = None if smap is None else np.asarray(smap, dtype=np.intp)
        fmap = None if fmap is None else np.asarray(fmap, dtype=np.intp)

        def _pad_1d(idx, vec):
            padded = np.zeros(tgt, dtype=vec.dtype)
            padded[idx] = vec
            return padded

        def _pad_2d(idx, mat):
            padded = np.zeros((mat.shape[0], tgt), dtype=mat.dtype)
            padded[:, idx] = mat
            return padded

        if u_loc.ndim == 1:
            if fmap is not None and fmap.shape[0] == u_loc.shape[0]:
                return _pad_1d(fmap, u_loc)
            if smap is not None and smap.shape[0] == u_loc.shape[0]:
                return _pad_1d(smap, u_loc)
            raise ValueError(
                f"Cannot pad coeffs (1D) for '{field_name}' on side {side}: "
                f"len(u_loc)={u_loc.shape[0]}, len(side_map)={None if smap is None else smap.shape[0]}, "
                f"len(field_map)={None if fmap is None else fmap.shape[0]}, target={tgt}"
            )

        elif u_loc.ndim == 2:
            if fmap is not None and fmap.shape[0] == u_loc.shape[1]:
                return _pad_2d(fmap, u_loc)
            if smap is not None and smap.shape[0] == u_loc.shape[1]:
                return _pad_2d(smap, u_loc)
            raise ValueError(
                f"Cannot pad coeffs (2D) for '{field_name}' on side {side}: "
                f"u_loc.shape[1]={'None' if u_loc.ndim < 2 else u_loc.shape[1]}, "
                f"len(side_map)={None if smap is None else smap.shape[0]}, "
                f"len(field_map)={None if fmap is None else fmap.shape[0]}, target={tgt}"
            )

        else:
            raise ValueError(f"Unsupported coeffs ndim={u_loc.ndim} for field '{field_name}'")


    @staticmethod
    def _align_phi_and_coeffs_to_global(ctx, side, field_name, phi, u_loc):
        """
        Convenience: returns (phi_aligned, u_aligned), both in the same global layout
        if ctx['global_dofs'] is present; otherwise returns inputs unchanged.
        """
        tgt = HelpersAlignCoefficents._target_len_from_ctx(ctx)
        if tgt is None:
            return phi, u_loc
        phi_g = HelpersAlignCoefficents._pad_basis_to_global(ctx, side, field_name, phi)
        u_g   = HelpersAlignCoefficents._pad_coeffs_to_global(ctx, side, field_name, u_loc)
        # Apply optional union masks (used by ghost integrals to enforce owner-side DOFs)
        mask_dict = ctx.get('pos_union_mask_by_field' if side == '+' else 'neg_union_mask_by_field')
        mask = None
        if isinstance(mask_dict, dict):
            mask = mask_dict.get(field_name)
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=u_g.dtype)
            if phi_g.ndim == 1:
                phi_g = phi_g * mask_arr
            else:
                phi_g = phi_g * mask_arr[np.newaxis, :]
            if u_g.ndim == 1:
                u_g = u_g * mask_arr
            else:
                u_g = u_g * mask_arr[np.newaxis, :]
        # Final sanity: shapes must match along the dof axis
        if phi_g.ndim == 1 and u_g.ndim == 1:
            assert phi_g.shape[0] == u_g.shape[0], "Basis/coeff size mismatch after padding"
        elif phi_g.ndim == 2 and u_g.ndim == 2:
            assert phi_g.shape[1] == u_g.shape[1], "Basis/coeff width mismatch after padding"
        elif phi_g.ndim == 2 and u_g.ndim == 1:
            assert phi_g.shape[1] == u_g.shape[0], "Basis width vs coeff length mismatch after padding"
        elif phi_g.ndim == 1 and u_g.ndim == 2:
            assert phi_g.shape[0] == u_g.shape[1], "Basis length vs coeff width mismatch after padding"
        return phi_g, u_g


# helpers.py  — selection normalizers used by DofHandler and FormCompiler


def normalize_elem_ids(mesh, elem_sel):
    """
    Normalize a variety of 'element selection' encodings into a list[int].

    Accepts
    -------
    - BitSet (with .to_indices() or .array)
    - str         : element bitset name on mesh (e.g., 'inside','outside','cut')
    - 1D bool mask
    - Iterable[int] (list/tuple/set/ndarray)
    - int         : single element id
    - None        : returns None

    Returns
    -------
    list[int] | None
    """
    if elem_sel is None:
        return None

    # single id (fast path)
    if isinstance(elem_sel, (int, np.integer)):
        return [int(elem_sel)]

    # BitSet
    if hasattr(elem_sel, "to_indices") and callable(getattr(elem_sel, "to_indices")):
        return [int(i) for i in elem_sel.to_indices()]
    if hasattr(elem_sel, "array"):
        arr = np.asarray(elem_sel.array, dtype=bool)
        return [int(i) for i in np.nonzero(arr)[0]]

    # named bitset on mesh
    if isinstance(elem_sel, str):
        bs = mesh.element_bitset(elem_sel)
        if hasattr(bs, "to_indices") and callable(getattr(bs, "to_indices")):
            return [int(i) for i in bs.to_indices()]
        arr = np.asarray(getattr(bs, "array", []), dtype=bool)
        return [int(i) for i in np.nonzero(arr)[0]]

    # iterable ids or boolean mask
    try:
        seq = elem_sel if isinstance(elem_sel, np.ndarray) else list(elem_sel)
    except TypeError:
        arr = np.asarray(elem_sel)
    else:
        arr = np.asarray(seq)

    if arr.dtype == bool:
        return [int(i) for i in np.nonzero(arr)[0]]
    if arr.ndim == 1:
        return [int(i) for i in arr.tolist()]

    raise ValueError("Unsupported element selection type.")

def normalize_edge_ids(mesh, edge_sel):
    """
    Normalize 'edge selection' into a list[int].

    Accepts BitSet (to_indices/array), str tag (mesh.edge_bitset),
    1D bool mask, iterable[int], int, or None.
    """
    if edge_sel is None:
        return None
    import numpy as np

    if isinstance(edge_sel, (int, np.integer)):
        return [int(edge_sel)]

    if hasattr(edge_sel, "to_indices") and callable(getattr(edge_sel, "to_indices")):
        return [int(i) for i in edge_sel.to_indices()]
    if hasattr(edge_sel, "array"):
        arr = np.asarray(edge_sel.array, dtype=bool)
        return [int(i) for i in np.nonzero(arr)[0]]

    if isinstance(edge_sel, str):
        bs = mesh.edge_bitset(edge_sel)
        if hasattr(bs, "to_indices") and callable(getattr(bs, "to_indices")):
            return [int(i) for i in bs.to_indices()]
        arr = np.asarray(getattr(bs, "array", []), dtype=bool)
        return [int(i) for i in np.nonzero(arr)[0]]

    try:
        seq = edge_sel if isinstance(edge_sel, np.ndarray) else list(edge_sel)
    except TypeError:
        arr = np.asarray(edge_sel)
    else:
        arr = np.asarray(seq)

    if arr.dtype == bool:
        return [int(i) for i in np.nonzero(arr)[0]]
    if arr.ndim == 1:
        return [int(i) for i in arr.tolist()]

    raise ValueError("Unsupported edge selection type.")
