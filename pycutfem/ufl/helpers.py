import numpy as np
from dataclasses import dataclass
from typing import Union, Tuple
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

    def inner(self, other: "GradOpInfo") -> np.ndarray:
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
