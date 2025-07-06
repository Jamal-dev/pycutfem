# --------------------------------------------------------------------
#  Robust triangulation for *any* field in a MixedElement space
# --------------------------------------------------------------------
from __future__ import annotations
import numpy as np
import matplotlib.tri as mtri
from typing import List

def triangulate_field(mesh, dof_handler, field: str, *, strict: bool = False) -> mtri.Triangulation:
    """
    Build a Matplotlib Triangulation that matches the DOFs of ``field``.

    * Works for triangles **and** quads, any polynomial order.
    * Safe for mixed-order spaces (e.g. Q2-Q1): elements whose corner node
      is **not** a DOF of the field are skipped or (if *strict*) raise.

    Parameters
    ----------
    mesh        : pycutfem.core.mesh.Mesh
    dof_handler : pycutfem.core.dofhandler.DofHandler
    field       : str        – field name (e.g. 'ux', 'p')
    strict      : bool       – raise instead of skipping incomplete elements
    """
    # 1.  Global DOFs and their coordinates in *field order*
    gdofs   = dof_handler.get_field_slice(field)                     # list[int]  :contentReference[oaicite:0]{index=0}
    node_ids: List[int] = [dof_handler._dof_to_node_map[d][1] for d in gdofs]  # mapping stored during build :contentReference[oaicite:1]{index=1}
    coords  = mesh.nodes_x_y_pos[node_ids]                           # shape (n,2)
    x, y    = coords[:, 0], coords[:, 1]

    # helper: map physical node-id → local index (0…n-1)
    id2local = {nid: i for i, nid in enumerate(node_ids)}

    # 2.  Element-wise connectivity (remapped to local indices)
    tris = []
    if mesh.element_type == "tri":
        for tri in mesh.corner_connectivity:
            try:
                tris.append([id2local[tri[0]], id2local[tri[1]], id2local[tri[2]]])
            except KeyError:
                if strict: raise ValueError("Field missing a triangle corner node.")
                continue
    elif mesh.element_type == "quad":
        for quad in mesh.corner_connectivity:
            try:
                # split into two triangles (0-1-3, 1-2-3)
                tris.append([id2local[quad[0]], id2local[quad[1]], id2local[quad[3]]])
                tris.append([id2local[quad[1]], id2local[quad[2]], id2local[quad[3]]])
            except KeyError:
                if strict: raise ValueError("Field missing a quadrilateral corner node.")
                continue
    else:
        raise KeyError(f"Unsupported element_type '{mesh.element_type}'")

    # 3.  Fallback: if *many* elements were skipped, rely on Delaunay
    if len(tris) < max(1, len(coords) // 5):        # heuristic – tweak if desired
        return mtri.Triangulation(x, y)

    return mtri.Triangulation(x, y, np.asarray(tris, dtype=int))
