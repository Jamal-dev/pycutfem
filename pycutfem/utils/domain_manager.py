import numpy as np
from pycutfem.core.mesh import Mesh
from pycutfem.utils.bitset import BitSet


def get_domain_bitset(mesh, entity_type: str, tag) -> BitSet:
    """
    Backward‑compatible helper that returns a BitSet for elements/edges.

    It now delegates to Mesh.get_domain_bitset / edge_bitset so that
    aggregate tags like 'ghost' work (covers ghost_pos/neg/both).
    """
    # Pass-through if the caller already gives a BitSet
    if isinstance(tag, BitSet):
        return tag

    kind = entity_type.lower()
    # Prefer Mesh API (uses O(1) caches created during classification)
    if hasattr(mesh, "get_domain_bitset"):
        entity = "edge" if kind.startswith("edge") else "element"
        return mesh.get_domain_bitset(tag, entity=entity)   # uses cached unions.  :contentReference[oaicite:11]{index=11}

    # Fallback – compute on the fly without touching Mesh caches
    if kind.startswith("edge"):
        if tag == "ghost":
            # accept any ghost_* tag on the edge
            mask = np.fromiter(((getattr(e, "tag", "") or "").startswith("ghost")
                                for e in mesh.edges_list), dtype=bool)
        else:
            mask = np.fromiter(((getattr(e, "tag", "") or "") == tag
                                for e in mesh.edges_list), dtype=bool)
    elif kind.startswith("elem"):
        mask = np.fromiter(((getattr(el, "tag", "") or "") == tag
                            for el in mesh.elements_list), dtype=bool)
    else:
        raise ValueError("entity_type must be 'element' or 'edge'")
    return BitSet(mask)
