# pycutfem/jit/ir.py
from dataclasses import dataclass, field
from typing import Union, Tuple, Callable, Any, Optional, List

# --- IR Node Definitions ---

@dataclass(frozen=True, slots=True)
class LoadVariable:
    """Instruction to load a variable (e.g., basis function, function coeffs)."""
    name: str          # Name of the symbolic variable (e.g., 'v', 'u_k')
    role: str          # 'test', 'trial', 'function'
    is_vector: bool    # Flag for vector-valued functions
    deriv_order: Tuple[int, int] = field(default=(0, 0)) # (dx, dy) order of derivative
    field_names: list = field(default_factory=list) 
    side: str = ""           # "", "+", or "-"   ← NEW
    field_sides: Optional[List[str]] = None

@dataclass(frozen=True, slots=True)
class LoadConstant:
    """Instruction to load a literal constant value."""
    value: Union[float, int]

@dataclass(frozen=True, slots=True)
class LoadElementWiseConstant:
    """Instruction to load a value from a per-element data array."""
    name: str          # An identifier for the data array
    tensor_shape: tuple[int, ...] = field(default=())

@dataclass(frozen=True, slots=True)
class LoadAnalytic:
    """Instruction to evaluate an analytic Python function."""
    # The callable itself cannot be pickled/hashed for caching.
    # We store its unique ID and will pass the actual function at runtime.
    func_id: int = field(compare=False)
    func_ref: Callable[[Any], Any] = field(repr=False, compare=False)
    tensor_shape: tuple[int, ...] = field(default=()) 

@dataclass(frozen=True, slots=True)
class LoadFacetNormal:
    """Instruction to load the normal vector at an interface quadrature point."""
    pass

@dataclass(frozen=True, slots =True)
class Grad:
    """Instruction to compute the gradient of the top of the stack."""
    pass

@dataclass(frozen=True, slots =True)
class Div:
    """Instruction to compute the divergence of the top of the stack."""
    pass

@dataclass(frozen=True, slots =True)
class PosOp:
    """Instruction to apply the positive-side restriction: v if phi>=0 else 0."""
    pass

@dataclass(frozen=True, slots =True)
class NegOp:
    """Instruction to apply the negative-side restriction: v if phi<0 else 0."""
    pass

@dataclass(frozen=True, slots =True)
class BinaryOp:
    """Instruction to perform a binary operation (e.g., +, *)."""
    op_symbol: str     # The operation symbol, e.g., '+', '*'

@dataclass(frozen=True, slots =True)
class Inner:
    """Instruction to compute the inner product of the top two stack items."""
    pass

@dataclass(frozen=True, slots =True)
class Dot:
    """Instruction to compute the dot product of the top two stack items."""
    pass

@dataclass(frozen=True, slots =True)
class Store:
    """Instruction to store the final result to an accumulator."""
    dest_name: str     # Name of the destination ('Ke' for matrix, 'Fe' for vector) or **'Je'**
    store_type: str    # 'vector', 'matrix' or 'functional'

@dataclass(frozen=True, slots =True)
class LoadConstantArray:
    """Instruction to load a constant array passed as a kernel argument."""
    name: str # A unique identifier for the constant array
    shape: Tuple[int, ...] # Shape of the constant array

@dataclass(frozen=True, slots=True)
class Transpose:
    """IR instruction: transpose top-of-stack tensor."""
    pass

@dataclass(frozen=True, slots=True)
class CellDiameter:
    pass

@dataclass(frozen=True, slots=True)
class LoadFacetNormalComponent:
    """Load n[idx] (scalar component of the facet normal)."""
    idx: int

@dataclass(frozen=True, slots=True)
class CheckDomain:
    """Instruction to check if the current element is in a domain BitSet."""
    bitset_id: int

@dataclass(frozen=True, slots=True)
class Trace:
    """Instruction to compute the trace of the tensor on top of the stack."""
    pass

@dataclass(frozen=True, slots=True)
class Hessian:   pass

@dataclass(frozen=True, slots=True)
class Laplacian: pass