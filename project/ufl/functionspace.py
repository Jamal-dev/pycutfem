import numpy as np
from pycutfem.fem.reference import get_reference


class FunctionSpace:
    """
    Represents a finite element function space.
    
    Attributes:
        mesh: The mesh object.
        p: The polynomial degree.
        dim: The dimension of the function (1 for scalar, 2 for 2D vector, etc.).
    """
    def __init__(self, mesh, p, dim=1):
        self.mesh = mesh
        self.p = p
        self.dim = dim

    def num_local_dofs(self):
        """Calculates the number of local DOFs for one element for a SCALAR field."""
        ref = get_reference(self.mesh.element_type, self.p)
        return len(ref.shape(0, 0)[0])

    def num_element_dofs(self):
        """Total number of DOFs on a single element for all components."""
        return self.num_local_dofs() * self.dim

    def num_global_dofs(self):
        """Total number of DOFs in the entire mesh."""
        return self.num_element_dofs() * len(self.mesh.elements_list)
