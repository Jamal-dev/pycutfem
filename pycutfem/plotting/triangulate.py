# --------------------------------------------------------------------
#  Robust triangulation for *any* field in a MixedElement space
# --------------------------------------------------------------------
from __future__ import annotations
import numpy as np
import matplotlib.tri as mtri
from typing import List
from itertools import chain



# triangulate.py  – robust per‑field connectivity


def triangulate_field(mesh, dof_handler, field: str, *, strict=False):
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
    # ------------------------------------------------------------------
    # 1.  Coordinates of *all* DOFs that belong to <field>
    # ------------------------------------------------------------------
    gdofs = dof_handler.get_field_slice(field)
    coords = dof_handler.get_dof_coords(field)
    x, y = coords[:, 0], coords[:, 1]

    dof2local = {int(g): i for i, g in enumerate(gdofs)}
    tris     = []

    # ------------------------------------------------------------------
    # 2.  Element‑wise triangulation that respects field DOFs
    # ------------------------------------------------------------------
    elem_maps = dof_handler.element_maps[field]
    for eid, edofs in enumerate(elem_maps):
        if len(edofs) < 3:
            if strict: raise ValueError(f"Element {eid} has <3 DOFs for '{field}'.")
            continue

        try:
            local_ids = [dof2local[int(d)] for d in edofs]
        except KeyError:
            if strict:
                missing = [int(d) for d in edofs if int(d) not in dof2local]
                raise ValueError(f"Element {eid} has DOFs {missing} not present in field '{field}'.")
            continue
        sub_coords  = coords[local_ids]

        # local Delaunay (matplotlib already has a C implementation)
        sub_tri = mtri.Triangulation(sub_coords[:,0], sub_coords[:,1])
        tris.extend([[local_ids[i] for i in tri]            # lift to global
                     for tri in sub_tri.triangles])

    # ------------------------------------------------------------------
    # 3.  Deduplicate and hand off to Matplotlib
    # ------------------------------------------------------------------
    if not tris:
        raise RuntimeError(f"Could not build connectivity for field '{field}'.")
    tris_arr = np.unique(np.sort(np.asarray(tris, dtype=int), axis=1), axis=0)
    return mtri.Triangulation(x, y, tris_arr)



def _extract_profile_1d(field_name: str,
                        dof_handler,
                        values: np.ndarray,
                        *,
                        line_axis: str,
                        line_pos: float,
                        atol: float = 1e-10):
    """
    Collect <values> of a scalar field along either
        - the vertical   line x = line_pos   (line_axis='x'),  or
        - the horizontal line y = line_pos   (line_axis='y').

    Repeated coordinates (may happen with high-order elements) are
    grouped and averaged so the returned arrays contain unique, sorted
    locations and one value per location.
    """
    # 1. coordinates of *all* DOFs of this field
    coords = dof_handler.get_dof_coords(field_name)        # shape (n_dofs,2)

    if line_axis.lower() == "x":         # vertical centre-line
        mask = np.isclose(coords[:, 0], line_pos, atol=atol)
        line_coord = coords[mask, 1]     # y-coordinates
    elif line_axis.lower() == "y":       # horizontal centre-line
        mask = np.isclose(coords[:, 1], line_pos, atol=atol)
        line_coord = coords[mask, 0]     # x-coordinates
    else:
        raise ValueError("line_axis must be 'x' or 'y'.")

    vals_on_line = values[mask]

    # 2. group (coord,value) pairs that are numerically identical
    #    → one representative per coordinate
    unique_map = {}          # coord -> list of values
    for c, v in zip(line_coord, vals_on_line):
        key = float(f"{c:.12f}")         # round to 1e-12 for stability
        unique_map.setdefault(key, []).append(v)

    sorted_items = sorted(unique_map.items())      # sort by coord
    coords_out = np.array([k for k, _ in sorted_items])
    vals_out   = np.array([np.mean(vs) for _, vs in sorted_items])
    return coords_out, vals_out
