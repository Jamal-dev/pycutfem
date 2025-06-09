from ufl.expressions import Integral

class Measure:
    """Represents an integration domain (dx, ds, dS)."""
    def __init__(self, domain_type, subdomain_tag=None):
        self.domain_type = domain_type
        self.subdomain_tag = subdomain_tag
        
    def __rmul__(self, other):
        """This is called for `expression * dx`."""
        return Integral(other, self)

def dx(subdomain_tag=None): return Measure("volume", subdomain_tag)
def ds(subdomain_tag=None): return Measure("interior_facet", subdomain_tag)
def dS(subdomain_tag=None): return Measure("exterior_facet", subdomain_tag)
