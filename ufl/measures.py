from ufl.expressions import Integral
from typing import Optional, Callable
from pycutfem.utils.bitset import BitSet # Import the BitSet class
class Measure:
    """
    Represents an integration domain (dx, ds), now controlled by BitSet objects
    and an optional level set for interface orientation.
    """
    def __init__(self, domain_type: str, defined_on: Optional[BitSet] = None, level_set: Optional[Callable] = None):
        self.domain_type = domain_type
        self.defined_on = defined_on
        self.level_set = level_set
        
    def __rmul__(self, other):
        """This is called for `expression * dx`."""
        return Integral(other, self)

def dx(defined_on: Optional[BitSet] = None): 
    """Creates a measure for integration over element volumes."""
    return Measure("volume", defined_on=defined_on)

def ds(defined_on: Optional[BitSet] = None, level_set: Optional[Callable] = None): 
    """Creates a measure for integration over interior facets (edges)."""
    return Measure("interior_facet", defined_on=defined_on, level_set=level_set)
def dS(defined_on: Optional[BitSet] = None, level_set: Optional[Callable] = None): 
    """Creates a measure for integration over exterior facets (edges)."""
    return Measure("exterior_facet", defined_on=defined_on, level_set=level_set)
