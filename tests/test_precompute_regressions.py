import numpy as np


def _slow_side_masks_by_field(dh, fields, eid: int, level_set, tol: float):
    """
    Reference implementation mirroring the historical logic in
    HelpersFieldAware.build_side_masks_by_field: evaluate φ at physical DOF
    coordinates using generic `phi_eval`, then apply SIDE convention and mark
    interface DOFs as belonging to both sides.
    """
    from pycutfem.core.sideconvention import SIDE
    from pycutfem.ufl.helpers_geom import phi_eval

    coords = dh.get_all_dof_coords()
    pos_masks = {}
    neg_masks = {}
    for fld in fields:
        try:
            gidx = np.asarray(dh.element_maps[fld][int(eid)], dtype=int)
        except Exception:
            continue
        xy = coords[gidx]
        phi = np.asarray([phi_eval(level_set, p) for p in xy], dtype=float)
        pos_mask = np.array([1.0 if SIDE.is_pos(val, tol=float(tol)) else 0.0 for val in phi], dtype=float)
        neg_mask = np.array([1.0 if SIDE.is_neg(val, tol=float(tol)) else 0.0 for val in phi], dtype=float)
        interface = np.abs(phi) <= float(tol)
        if np.any(interface):
            pos_mask[interface] = 1.0
            neg_mask[interface] = 1.0
        pos_masks[fld] = pos_mask
        neg_masks[fld] = neg_mask
    return pos_masks, neg_masks


def test_hash_subset_handles_large_indices():
    from pycutfem.core.dofhandler import _hash_subset

    # Mimic interface-segment packing (eid<<16)|sid on a large mesh.
    large_eid = 40000
    packed = (int(large_eid) << 16) ^ 7
    h = _hash_subset([packed, packed + 1, 123])
    assert isinstance(h, int)


def test_side_masks_fastpath_matches_physical_eval():
    from pycutfem.core.mesh import Mesh
    from pycutfem.core.levelset import LevelSetGridFunction
    from pycutfem.core.dofhandler import DofHandler
    from pycutfem.fem.mixedelement import MixedElement
    from pycutfem.utils.meshgen import structured_quad
    from pycutfem.ufl.helpers import HelpersFieldAware
    from pycutfem.core.sideconvention import SIDE

    poly_order = 2
    nodes, elements, edges, corners = structured_quad(1.0, 1.0, nx=4, ny=3, poly_order=poly_order)
    mesh = Mesh(
        nodes=nodes,
        element_connectivity=elements,
        edges_connectivity=edges,
        elements_corner_nodes=corners,
        element_type="quad",
        poly_order=poly_order,
    )

    # Unknowns live on the same mesh, with mixed orders.
    dh = DofHandler(MixedElement(mesh, field_specs={"u": 2, "p": 1}), method="cg")

    # FE-backed level set on its own scalar space.
    ls_dh = DofHandler(MixedElement(mesh, field_specs={"phi": 2}), method="cg")
    level_set = LevelSetGridFunction(ls_dh, field="phi")
    level_set.interpolate(lambda x, y: float(x - 0.48))  # produces both + and - DOFs
    level_set.commit(tol=1e-12)

    fields = tuple(dh.mixed_element.field_names)
    eid = 0

    pos_fast, neg_fast = HelpersFieldAware.build_side_masks_by_field(
        dh, fields, eid, level_set, tol=float(SIDE.tol)
    )
    pos_slow, neg_slow = _slow_side_masks_by_field(dh, fields, eid, level_set, tol=float(SIDE.tol))

    assert pos_fast.keys() == pos_slow.keys()
    assert neg_fast.keys() == neg_slow.keys()
    for fld in fields:
        assert np.array_equal(pos_fast[fld], pos_slow[fld]), fld
        assert np.array_equal(neg_fast[fld], neg_slow[fld]), fld

