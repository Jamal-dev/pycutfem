# pycutfem/ufl/measures.py
from __future__ import annotations
from typing import Optional, Callable, Dict

# Import Integral for type hinting and __rmul__
from pycutfem.ufl.expressions import Integral, Expression 
# Assuming BitSet is in utils. We can use a forward reference if needed.
from pycutfem.utils.bitset import BitSet 

class Measure:
    """
    Represents an integration measure (dx, ds, etc.) with support for
    domain specification (via BitSet and level sets for CutFEM) and
    numerical metadata (like quadrature degree).
    """
    def __init__(self, 
                 domain_type: str, 
                 defined_on: Optional[BitSet] = None, 
                 level_set: Optional[Callable] = None,
                 metadata: Optional[Dict] = None,
                 tag: Optional[str] = None,
                 deformation = None):
        """
        Initializes a Measure.
        
        Args:
            domain_type: The type of domain ('volume', 'interior_facet', etc.).
            defined_on: A BitSet specifying the subset of elements or facets
                        this measure applies to.
            level_set: A level-set function, used for orientation in
                       interface and cut-cell integrals.
            metadata: A dictionary for backend-specific information, such as
                      {'quad_degree': 5}.
        """
        self.domain_type = domain_type
        self.defined_on = defined_on
        self.level_set = level_set
        self.metadata = metadata.copy() if metadata else {}
        self.deformation = deformation
        if deformation is not None:
            self.metadata.setdefault('deformation', deformation)
        self.tag = tag
    @property
    def on_facet(self) -> bool:
        return self.domain_type in {
            "interior_facet", "exterior_facet", "interface", "ghost_edge"
        }

    def __call__(self, 
                 defined_on: Optional[BitSet] = None, 
                 level_set: Optional[Callable] = None,
                 metadata: Optional[Dict] = None,
                 tag: Optional[str] = None,
                 deformation = None) -> 'Measure':
        """
        Creates a new, configured Measure instance. This allows for an
        expressive syntax like `dx(defined_on=my_set, metadata={'q': 5})`.
        """
        # Create a new metadata dictionary, starting with existing values
        new_meta = self.metadata.copy()
        if metadata:
            new_meta.update(metadata)
        
        # Use new values if provided, otherwise stick with the defaults
        new_defined_on = defined_on if defined_on is not None else self.defined_on
        new_level_set = level_set if level_set is not None else self.level_set
        new_tag = tag if tag is not None else self.tag
        new_deformation = deformation if deformation is not None else self.deformation
        # Return a new Measure object with the specified configurations
        return Measure(self.domain_type, new_defined_on, new_level_set, new_meta, new_tag, new_deformation)

    def __rmul__(self, other: Expression) -> Integral:
        """
        Handles the `integrand * dx` operation, creating an Integral.
        """
        return Integral(other, self)

    def __repr__(self):
        """Provides a clear string representation of the Measure."""
        parts = [f"type='{self.domain_type}'"]
        if self.defined_on:
            parts.append(f"defined_on={self.defined_on!r}")
        if self.level_set:
            parts.append(f"level_set={self.level_set.__name__ if hasattr(self.level_set, '__name__') else '...'}")
        if self.tag:        parts.append(f"tag='{self.tag}'") 
        if self.metadata:
            parts.append(f"metadata={self.metadata!r}")
        if self.deformation is not None:
            parts.append("deformation=...")
        return f"Measure({', '.join(parts)})"

# --- Pre-defined global instances for convenient use in weak forms ---

#: Measure for integration over element volumes.
dx = Measure("volume")

#: Measure for integration over interior facets (edges).
ds = Measure("interior_facet")

#: Measure for integration over exterior (boundary) facets (edges).
dS = Measure("exterior_facet")

#: Measure for integration over a level-set-defined interface.
dInterface = Measure("interface")

# New: Measure for integration over ghost edges for CutFEM stabilization.
dGhost = Measure("ghost_edge")