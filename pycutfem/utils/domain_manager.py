import numpy as np

from pycutfem.utils.bitset import BitSet

def get_domain_bitset(mesh: "Mesh", entity_type: str, tag: str) -> BitSet:
    """
    Return a BitSet that selects all *entity_type* (‘element’ or ‘edge’) whose
    `.tag` equals *tag*.  For interface/ghost problems the routine first tries
    the BitSet caches filled during `mesh.classify_elements/edges`, falling
    back to an explicit O(n) scan only when necessary.
    """
    if entity_type == "edge":
        # Fast path – use cache built once in classify_edges(..)
        if tag in getattr(mesh, "_edge_bitsets", {}):
            return mesh.edge_bitset(tag)                     # O(1)
        # ------------------------------------------------------------------
        entities = mesh.edges_list
    elif entity_type == "element":
        if tag in getattr(mesh, "_elem_bitsets", {}):
            return mesh.element_bitset(tag)
        entities = mesh.elements_list
    else:
        raise ValueError("entity_type must be 'element' or 'edge'")

    # Slow fall-back – happens only if the mesh has not been classified yet
    mask = np.fromiter((getattr(e, "tag", "") == tag for e in entities),
                       dtype=bool, count=len(entities))
    bs = BitSet(mask)
    # Fill the cache so subsequent calls are cheap
    if entity_type == "edge":
        mesh._edge_bitsets[tag] = bs
    else:
        mesh._elem_bitsets[tag] = bs
    return bs
