import numpy as np
from pycutfem.core.mesh import Mesh
from pycutfem.utils.bitset import BitSet

def get_domain_bitset(mesh: Mesh, entity_type: str, tag: str) -> BitSet:
    """
    Creates a BitSet for a domain based on a tag applied to mesh entities.

    Args:
        mesh (Mesh): The mesh object, which must have been classified first.
        entity_type (str): The type of mesh entity ('element' or 'edge').
        tag (str): The tag to search for (e.g., 'NEG', 'POS', 'interface').

    Returns:
        BitSet: A bitmask representing the domain.
    """
    if entity_type == 'element':
        entity_list = mesh.elements_list
        mask = np.array([e.tag == tag for e in entity_list], dtype=bool)
    elif entity_type == 'edge':
        entity_list = mesh.edges_list
        mask = np.array([e.tag == tag for e in entity_list], dtype=bool)
    else:
        raise ValueError("entity_type must be 'element' or 'edge'")
        
    return BitSet(mask)