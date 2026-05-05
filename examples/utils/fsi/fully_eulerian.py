from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from pycutfem.core.geometry import hansbo_cut_ratio
from pycutfem.core.mesh import Mesh
from pycutfem.core.dofhandler import DofHandler
from pycutfem.utils.bitset import BitSet
from pycutfem.ufl.expressions import (
    CellDiameter,
    Constant,
    Derivative,
    FacetNormal,
    ElementWiseConstant,
    Pos,
    Neg,
    Hessian,
    grad,
    inner,
    dot,
    div,
    jump,
    Identity,
    det,
    inv,
    trace,
    restrict,
)
from pycutfem.ufl.measures import dx, dGhost, dInterface, dFacetPatch
from examples.utils.fsi.contact import sigma_s_stvk as sigma_s_stvk_paper, dsigma_s_stvk as dsigma_s_stvk_paper
from pycutfem.ufl.helpers import analyze_active_dofs


def _copy_bitset(bs: BitSet) -> BitSet:
    return BitSet(np.array(bs.mask, dtype=bool))

def _ghost_band_edges(mesh: Mesh, seed: BitSet, *, layers: int = 2) -> BitSet:
    """Return a ghost-edge BitSet from an element band around `seed`.

    The band is built by a BFS over element adjacency (`mesh.neighbors()`). We then
    mark all *interior* edges whose two owner elements lie in the band, excluding
    aligned interface edges.
    """
    layers = max(int(layers), 0)
    if seed.cardinality() == 0:
        return BitSet(np.zeros(len(mesh.edges_list), dtype=bool))

    seed_ids = seed.to_indices()
    band: set[int] = set(map(int, seed_ids))
    frontier: set[int] = set(band)
    nbs = mesh.neighbors()

    for _ in range(layers):
        if not frontier:
            break
        nxt: set[int] = set()
        for eid in frontier:
            for nb in nbs[int(eid)]:
                nb = int(nb)
                if nb not in band:
                    nxt.add(nb)
        band.update(nxt)
        frontier = nxt

    mask = np.zeros(len(mesh.edges_list), dtype=bool)
    for e in mesh.edges_list:
        if e.right is None:
            continue
        if int(e.left) in band and int(e.right) in band:
            mask[int(e.gid)] = True

    interface_bs = mesh.edge_bitset("interface")
    return BitSet(mask) - interface_bs


def _element_band(mesh: Mesh, seed: BitSet, *, layers: int = 1) -> BitSet:
    """Return an element BitSet expanded by `layers` of element adjacency."""
    layers = max(int(layers), 0)
    if layers == 0 or seed.cardinality() == 0:
        return _copy_bitset(seed)

    seed_ids = seed.to_indices()
    band: set[int] = set(map(int, seed_ids))
    frontier: set[int] = set(band)
    nbs = mesh.neighbors()

    for _ in range(layers):
        if not frontier:
            break
        nxt: set[int] = set()
        for eid in frontier:
            for nb in nbs[int(eid)]:
                nb = int(nb)
                if nb not in band:
                    nxt.add(nb)
        band.update(nxt)
        frontier = nxt

    mask = np.zeros(len(mesh.elements_list), dtype=bool)
    if band:
        ids = np.fromiter(band, dtype=int)
        mask[ids] = True
    return BitSet(mask)


def _extension_edges(
    mesh: Mesh,
    *,
    base_elems: BitSet,
    ext_elems: BitSet,
    exclude_edges: BitSet,
) -> BitSet:
    """
    Extension facet set (paper Eq. (14)):
      - interior edges with both owners in `ext_elems`
      - at least one owner in `ext_elems \\ base_elems`
      - excluding `exclude_edges` (e.g. ghost/interface facets).
    """
    if ext_elems.cardinality() == 0:
        return BitSet(np.zeros(len(mesh.edges_list), dtype=bool))
    ext_only = ext_elems - base_elems
    if ext_only.cardinality() == 0:
        return BitSet(np.zeros(len(mesh.edges_list), dtype=bool))

    ext_mask = np.asarray(ext_elems.mask, dtype=bool)
    ext_only_mask = np.asarray(ext_only.mask, dtype=bool)

    mask = np.zeros(len(mesh.edges_list), dtype=bool)
    for e in mesh.edges_list:
        if e.right is None:
            continue
        l = int(e.left)
        r = int(e.right)
        if ext_mask[l] and ext_mask[r] and (ext_only_mask[l] or ext_only_mask[r]):
            mask[int(e.gid)] = True

    bs = BitSet(mask)
    if exclude_edges.cardinality() > 0:
        bs = bs - exclude_edges
    # Always exclude the physical interface facets.
    interface_bs = mesh.edge_bitset("interface")
    return bs - interface_bs


def nudge_levelset_zeros(level_set, eps: float, *, prefer_negative: bool = True, commit: bool = True) -> int:
    """Nudge |phi|<=eps nodes away from 0 to avoid aligned-interface degeneracy."""
    if eps <= 0.0:
        return 0
    if not hasattr(level_set, "nodal_values"):
        raise TypeError("level_set must provide nodal_values() for nudging.")
    phi_vals = level_set.nodal_values()
    mask = np.abs(phi_vals) <= eps
    if not np.any(mask):
        return 0
    phi_vals[mask] = -eps if prefer_negative else eps
    if commit:
        if not hasattr(level_set, "commit"):
            raise TypeError("level_set must provide commit() when commit=True.")
        level_set.commit()
    return int(mask.sum())


def _aligned_cut_mask(mesh: Mesh) -> np.ndarray:
    """Flag cut elements whose interface lies exactly on an element edge."""
    try:
        interface_edges = mesh.edge_bitset("interface").to_indices()
    except Exception:
        interface_edges = np.array([], dtype=int)
    if interface_edges.size == 0:
        return np.zeros(len(mesh.elements_list), dtype=bool)
    interface_set = set(map(int, interface_edges))
    aligned = np.zeros(len(mesh.elements_list), dtype=bool)
    for elem in mesh.elements_list:
        if getattr(elem, "tag", "") != "cut":
            continue
        for side_edges in elem.edges_by_side:
            for gid in side_edges:
                if int(gid) in interface_set:
                    aligned[int(elem.id)] = True
                    break
            if aligned[int(elem.id)]:
                break
    return aligned


def make_domain_sets(
    mesh: Mesh,
    *,
    use_aligned_interface: bool = False,
    extension_layers: int = 0,
) -> Dict[str, BitSet]:
    fluid = mesh.element_bitset("outside")
    solid = mesh.element_bitset("inside")
    cut = mesh.element_bitset("cut")
    if use_aligned_interface and mesh.edge_bitset("interface").cardinality() > 0:
        aligned_cut = _aligned_cut_mask(mesh)
        cut_interface = BitSet(cut.mask & ~aligned_cut)
    else:
        cut_interface = _copy_bitset(cut)
    fluid_ifc = fluid | cut
    solid_ifc = solid | cut
    has_pos = fluid | cut
    has_neg = solid | cut

    interface_bs = mesh.edge_bitset("interface")
    ghost_pos = mesh.edge_bitset("ghost_pos") - interface_bs
    ghost_neg = mesh.edge_bitset("ghost_neg") - interface_bs
    ghost_both = mesh.edge_bitset("ghost_both") - interface_bs
    ghost_all = (ghost_pos | ghost_neg | ghost_both) - interface_bs

    solid_ghost = ghost_neg | ghost_both
    fluid_ghost = ghost_pos | ghost_both

    # Sliver/corner robustness:
    # If one side has no fully-inside/outside elements, cut cells can be supported
    # by only a handful of facets (or none), which can make stabilization too weak.
    # In that case, use a small element-band around the cut region and include all
    # interior edges within that band for ghost stabilization.
    if cut.cardinality() > 0:
        if fluid.cardinality() == 0 or solid.cardinality() == 0:
            band_ghost = _ghost_band_edges(mesh, cut, layers=2)
            if band_ghost.cardinality() > 0:
                fluid_ghost = _copy_bitset(band_ghost)
                solid_ghost = _copy_bitset(band_ghost)
        else:
            if fluid_ghost.cardinality() == 0 and ghost_all.cardinality() > 0:
                fluid_ghost = _copy_bitset(ghost_all)
            if solid_ghost.cardinality() == 0 and ghost_all.cardinality() > 0:
                solid_ghost = _copy_bitset(ghost_all)
    out = {
        "fluid_domain": _copy_bitset(fluid),
        "solid_domain": _copy_bitset(solid),
        "cut_domain": _copy_bitset(cut),
        "cut_interface": _copy_bitset(cut_interface),
        "fluid_interface": _copy_bitset(fluid_ifc),
        "solid_interface": _copy_bitset(solid_ifc),
        "has_pos": _copy_bitset(has_pos),
        "has_neg": _copy_bitset(has_neg),
        "solid_ghost": _copy_bitset(solid_ghost),
        "fluid_ghost": _copy_bitset(fluid_ghost),
    }

    ext_layers = max(int(extension_layers), 0)
    if ext_layers > 0:
        fluid_ext = _element_band(mesh, fluid_ifc, layers=ext_layers)
        solid_ext = _element_band(mesh, solid_ifc, layers=ext_layers)

        fluid_ext_edges = _extension_edges(
            mesh,
            base_elems=fluid_ifc,
            ext_elems=fluid_ext,
            exclude_edges=fluid_ghost,
        )
        solid_ext_edges = _extension_edges(
            mesh,
            base_elems=solid_ifc,
            ext_elems=solid_ext,
            exclude_edges=solid_ghost,
        )

        out.update(
            {
                "fluid_ext_domain": _copy_bitset(fluid_ext),
                "solid_ext_domain": _copy_bitset(solid_ext),
                "has_pos_ext": _copy_bitset(fluid_ext),
                "has_neg_ext": _copy_bitset(solid_ext),
                "fluid_ext_ghost": _copy_bitset(fluid_ext_edges),
                "solid_ext_ghost": _copy_bitset(solid_ext_edges),
            }
        )
    return out


def _update_bs(target: BitSet, new_mask: np.ndarray) -> None:
    target.mask[...] = np.asarray(new_mask, dtype=bool)


def refresh_domain_sets(
    mesh: Mesh,
    domains: Dict[str, BitSet],
    *,
    use_aligned_interface: bool = False,
    extension_layers: int = 0,
) -> None:
    """
    Update `domains` *in place* after re-classifying `mesh` against a (moving) level set.

    This is the moving-interface analogue of `make_domain_sets(...)`. It is designed
    so any previously created `restrict(...)` wrappers and Measures that captured
    the BitSet objects remain valid (their masks are updated, not replaced).
    """
    fluid = mesh.element_bitset("outside")
    solid = mesh.element_bitset("inside")
    cut = mesh.element_bitset("cut")
    has_fluid = fluid | cut
    has_solid = solid | cut

    if use_aligned_interface and mesh.edge_bitset("interface").cardinality() > 0:
        aligned_cut = _aligned_cut_mask(mesh)
        cut_interface_mask = cut.mask & ~aligned_cut
    else:
        cut_interface_mask = cut.mask

    _update_bs(domains["fluid_domain"], fluid.mask)
    _update_bs(domains["solid_domain"], solid.mask)
    _update_bs(domains["cut_domain"], cut.mask)
    _update_bs(domains["cut_interface"], cut_interface_mask)
    _update_bs(domains["fluid_interface"], has_fluid.mask)
    _update_bs(domains["solid_interface"], has_solid.mask)
    _update_bs(domains["has_pos"], fluid.mask | cut.mask)
    _update_bs(domains["has_neg"], solid.mask | cut.mask)

    interface_bs = mesh.edge_bitset("interface")
    ghost_pos = mesh.edge_bitset("ghost_pos") - interface_bs
    ghost_neg = mesh.edge_bitset("ghost_neg") - interface_bs
    ghost_both = mesh.edge_bitset("ghost_both") - interface_bs
    ghost_all = (ghost_pos | ghost_neg | ghost_both) - interface_bs

    solid_ghost = ghost_neg | ghost_both
    fluid_ghost = ghost_pos | ghost_both

    cut_now = mesh.element_bitset("cut")
    if cut_now.cardinality() > 0:
        if fluid.cardinality() == 0 or solid.cardinality() == 0:
            band_ghost = _ghost_band_edges(mesh, cut_now, layers=2)
            if band_ghost.cardinality() > 0:
                fluid_ghost = _copy_bitset(band_ghost)
                solid_ghost = _copy_bitset(band_ghost)
        else:
            if fluid_ghost.cardinality() == 0 and ghost_all.cardinality() > 0:
                fluid_ghost = _copy_bitset(ghost_all)
            if solid_ghost.cardinality() == 0 and ghost_all.cardinality() > 0:
                solid_ghost = _copy_bitset(ghost_all)

    _update_bs(domains["solid_ghost"], solid_ghost.mask)
    _update_bs(domains["fluid_ghost"], fluid_ghost.mask)

    # Optional implicit-extension sets (paper Sec. 3.4).
    ext_layers = max(int(extension_layers), 0)
    if ext_layers > 0 or ("fluid_ext_domain" in domains or "solid_ext_domain" in domains):
        ext_layers_eff = ext_layers if ext_layers > 0 else 1
        fluid_ext = _element_band(mesh, has_fluid, layers=ext_layers_eff)
        solid_ext = _element_band(mesh, has_solid, layers=ext_layers_eff)

        fluid_ext_edges = _extension_edges(
            mesh,
            base_elems=has_fluid,
            ext_elems=fluid_ext,
            exclude_edges=fluid_ghost,
        )
        solid_ext_edges = _extension_edges(
            mesh,
            base_elems=has_solid,
            ext_elems=solid_ext,
            exclude_edges=solid_ghost,
        )

        if "fluid_ext_domain" in domains:
            _update_bs(domains["fluid_ext_domain"], fluid_ext.mask)
        if "solid_ext_domain" in domains:
            _update_bs(domains["solid_ext_domain"], solid_ext.mask)
        if "has_pos_ext" in domains:
            _update_bs(domains["has_pos_ext"], fluid_ext.mask)
        if "has_neg_ext" in domains:
            _update_bs(domains["has_neg_ext"], solid_ext.mask)
        if "fluid_ext_ghost" in domains:
            _update_bs(domains["fluid_ext_ghost"], fluid_ext_edges.mask)
        if "solid_ext_ghost" in domains:
            _update_bs(domains["solid_ext_ghost"], solid_ext_edges.mask)


# Backwards-compatible alias (some examples use the shorter name).
refresh_domains = refresh_domain_sets


def build_measures(
    mesh: Mesh,
    level_set,
    domains: Dict[str, BitSet],
    qvol: int = 6,
    *,
    use_facet_patch_ghost: bool = True,
):
    dx_fluid = dx(
        defined_on=domains["fluid_interface"],
        level_set=level_set,
        metadata={"q": qvol, "side": "+"},
    )
    dx_solid = dx(
        defined_on=domains["solid_interface"],
        level_set=level_set,
        metadata={"q": qvol, "side": "-"},
    )
    dGamma = dInterface(
        defined_on=domains["cut_interface"],
        level_set=level_set,
        metadata={"q": qvol + 2, "derivs": {(0, 0), (0, 1), (1, 0)}},
    )
    ghost_measure = dFacetPatch if use_facet_patch_ghost else dGhost
    ghost_derivs = {(0, 1), (1, 0)}
    # Facet-patch kernels may need 2nd derivatives (Hessian ghost penalties).
    if use_facet_patch_ghost:
        ghost_derivs |= {(2, 0), (1, 1), (0, 2)}
    dG_fluid = ghost_measure(
        defined_on=domains["fluid_ghost"],
        level_set=level_set,
        metadata={"q": qvol, "derivs": ghost_derivs},
    )
    dG_solid = ghost_measure(
        defined_on=domains["solid_ghost"],
        level_set=level_set,
        metadata={"q": qvol, "derivs": ghost_derivs},
    )
    return dx_fluid, dx_solid, dGamma, dG_fluid, dG_solid


def build_extension_measures(
    mesh: Mesh,
    level_set,
    domains: Dict[str, BitSet],
    qvol: int = 6,
    *,
    use_facet_patch_ghost: bool = True,
):
    """
    Build additional ghost-type measures on the implicit extension facets
    (paper Eq. (14)).

    Requires `make_domain_sets(..., extension_layers>0)` or
    `refresh_domain_sets(..., extension_layers>0)` to populate:
      - domains["fluid_ext_ghost"]
      - domains["solid_ext_ghost"]
    """
    if "fluid_ext_ghost" not in domains or "solid_ext_ghost" not in domains:
        raise KeyError("Extension sets missing; call make_domain_sets(..., extension_layers>0) first.")

    ghost_measure = dFacetPatch if use_facet_patch_ghost else dGhost
    ghost_derivs = {(0, 1), (1, 0)}
    if use_facet_patch_ghost:
        ghost_derivs |= {(2, 0), (1, 1), (0, 2)}
    dG_fluid_ext = ghost_measure(
        defined_on=domains["fluid_ext_ghost"],
        level_set=level_set,
        metadata={"q": qvol, "derivs": ghost_derivs},
    )
    dG_solid_ext = ghost_measure(
        defined_on=domains["solid_ext_ghost"],
        level_set=level_set,
        metadata={"q": qvol, "derivs": ghost_derivs},
    )
    return dG_fluid_ext, dG_solid_ext


def hansbo_kappa(
    mesh: Mesh,
    level_set,
    *,
    theta_min: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray]:
    theta_pos_vals = np.clip(hansbo_cut_ratio(mesh, level_set, side="+"), theta_min, 1.0)
    theta_neg_vals = np.clip(hansbo_cut_ratio(mesh, level_set, side="-"), theta_min, 1.0)
    return theta_pos_vals, theta_neg_vals


def refresh_sliver_weights(
    mesh: Mesh,
    theta_pos_vals: np.ndarray,
    theta_neg_vals: np.ndarray,
    w_pos_vals: np.ndarray,
    w_neg_vals: np.ndarray,
    *,
    theta0: float = 0.05,
    p: float = 1.0,
    wmax: float = 1000.0,
    thetamin: float = 1.0e-6,
    smooth: float = 0.3,
) -> None:
    """
    Update per-element sliver weights based on Hansbo cut ratios.

    Weights are only amplified on cut elements with tiny theta to avoid
    blow-ups when a nearly aligned interface produces extreme slivers.
    """
    theta0 = max(float(theta0), float(thetamin))
    wmax = max(float(wmax), 1.0)
    cut_ids = mesh.element_bitset("cut").to_indices()
    target_pos = np.ones_like(w_pos_vals)
    target_neg = np.ones_like(w_neg_vals)

    if cut_ids.size:
        thp = np.maximum(theta_pos_vals[cut_ids], thetamin)
        thn = np.maximum(theta_neg_vals[cut_ids], thetamin)

        scale_p = np.ones_like(thp)
        mask_p = thp < theta0
        if np.any(mask_p):
            scale_p[mask_p] = np.minimum(wmax, (theta0 / thp[mask_p]) ** p)

        scale_n = np.ones_like(thn)
        mask_n = thn < theta0
        if np.any(mask_n):
            scale_n[mask_n] = np.minimum(wmax, (theta0 / thn[mask_n]) ** p)

        target_pos[cut_ids] = scale_p
        target_neg[cut_ids] = scale_n

    if smooth <= 0.0 or smooth >= 1.0:
        w_pos_vals[:] = target_pos
        w_neg_vals[:] = target_neg
    else:
        w_pos_vals[:] = (1.0 - smooth) * w_pos_vals + smooth * target_pos
        w_neg_vals[:] = (1.0 - smooth) * w_neg_vals + smooth * target_neg


def retag_inactive(
    dh: DofHandler,
    mesh: Mesh,
    *,
    theta_neg: Optional[np.ndarray] = None,
    solid_cut_drop: float = 0.0,
    fluid_ext_domain: Optional[BitSet] = None,
    solid_ext_domain: Optional[BitSet] = None,
) -> None:
    dh.dof_tags["inactive"] = set()
    inside_mask = np.asarray(mesh.element_bitset("inside").mask, dtype=bool)
    outside_mask = np.asarray(mesh.element_bitset("outside").mask, dtype=bool)

    if isinstance(fluid_ext_domain, BitSet) and fluid_ext_domain.cardinality() > 0:
        inside_mask = inside_mask & ~np.asarray(fluid_ext_domain.mask, dtype=bool)
    if isinstance(solid_ext_domain, BitSet) and solid_ext_domain.cardinality() > 0:
        outside_mask = outside_mask & ~np.asarray(solid_ext_domain.mask, dtype=bool)

    dh.tag_dofs_from_element_bitset("inactive", "u_pos_x", inside_mask, strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "u_pos_y", inside_mask, strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "p_pos_", inside_mask, strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "vs_neg_x", outside_mask, strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "vs_neg_y", outside_mask, strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "d_neg_x", outside_mask, strict=True)
    dh.tag_dofs_from_element_bitset("inactive", "d_neg_y", outside_mask, strict=True)
    if theta_neg is not None and solid_cut_drop > 0.0:
        cut_mask = mesh.element_bitset("cut").mask
        bad = cut_mask & (theta_neg < solid_cut_drop)
        if np.any(bad):
            for field in ("vs_neg_x", "vs_neg_y", "d_neg_x", "d_neg_y"):
                dh.tag_dofs_from_element_bitset("inactive", field, bad, strict=False)


def recompute_active_dofs(solver, bcs_active) -> bool:
    """
    Recompute and update `solver.active_dofs` after a moving-interface re-tagging.

    This mirrors the logic used during `NewtonSolver.__init__`:
      - honor `Restriction(...)` domains
      - drop Dirichlet DOFs
      - drop DOFs tagged as inactive
      - rebuild reduced-space maps and mark the reduced pattern stale
    """
    from pycutfem.solvers.nonlinear_solver import _ActiveReducer  # local import (private helper)

    dh = solver.dh
    constraints = getattr(solver, "constraints", None)
    ndof_eff = int(constraints.n_master) if constraints is not None else int(dh.total_dofs)

    old_active = np.asarray(getattr(solver, "active_dofs", np.empty(0, dtype=int)), dtype=int)
    active_by_restr, has_restriction = analyze_active_dofs(solver.equation, dh, solver.me, bcs_active, verbose=False)
    bc_dofs_full = set(dh.get_dirichlet_data(bcs_active).keys())
    inactive_full = set(getattr(dh, "dof_tags", {}).get("inactive", set()))
    inactive_free_full = inactive_full - bc_dofs_full

    if constraints is None:
        candidate = set(active_by_restr) if has_restriction else set(range(ndof_eff))
        free = sorted((candidate - bc_dofs_full) - inactive_free_full)
        new_active = np.asarray(free, dtype=int)
    else:
        candidate_master = constraints.to_master_set(active_by_restr) if has_restriction else set(range(ndof_eff))
        bc_master = constraints.to_master_set(bc_dofs_full)
        inactive_master = constraints.to_master_set(inactive_free_full)
        free = sorted((candidate_master - bc_master) - inactive_master)
        new_active = np.asarray(free, dtype=int)

    dec_mask = getattr(solver, "_decoupled_full_mask", None)
    if isinstance(dec_mask, np.ndarray) and dec_mask.dtype == bool and int(dec_mask.size) == int(ndof_eff):
        if new_active.size:
            new_active = new_active[~dec_mask[new_active]]

    if old_active.size == new_active.size and np.array_equal(old_active, new_active):
        return False

    solver.active_dofs = new_active
    solver.full_to_red = -np.ones(ndof_eff, dtype=int)
    solver.full_to_red[solver.active_dofs] = np.arange(len(solver.active_dofs), dtype=int)
    solver.red_to_full = solver.active_dofs
    solver.use_reduced = len(solver.active_dofs) < ndof_eff
    solver.restrictor = _ActiveReducer(dh, solver.active_dofs, constraint=constraints)
    solver._pattern_stale = True
    return True


def extend_newly_active_dofs_nearest(
    *,
    dh: DofHandler,
    newly_active: np.ndarray,
    active_old: np.ndarray,
    active_new: np.ndarray,
    field_to_current,
    field_to_prev,
    k: int = 4,
    trace: bool = False,
) -> None:
    """
    Initialize newly-active DOFs by a k-NN inverse-distance extension.

    This reduces residual spikes when the interface moves and previously inactive
    DOFs become part of the solve.
    """
    newly_active = np.asarray(newly_active, dtype=int).ravel()
    if newly_active.size == 0:
        return

    active_new = np.asarray(active_new, dtype=int).ravel()
    stable_active = np.setdiff1d(active_new, newly_active, assume_unique=False)
    if stable_active.size == 0:
        return

    try:
        dh._ensure_dof_coords()  # internal but stable
        coords = np.asarray(dh._dof_coords, dtype=float)
    except Exception:
        return

    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception:
        return

    field_names = sorted(set(field_to_current.keys()) & set(field_to_prev.keys()))
    if not field_names:
        return

    field_slices: dict[str, np.ndarray] = {f: np.asarray(dh.get_field_slice(f), dtype=int) for f in field_names}

    def _carrier_field_data(carrier, field: str) -> tuple[np.ndarray, dict[int, int]] | None:
        """
        Return (nodal_values_view, g2l_map) for a single scalar field.

        Note: VectorFunction carriers store all component fields in a single
        concatenated array (`carrier.nodal_values`) indexed by `carrier._g2l`
        (global dof -> local index). For extension we operate on the parent
        storage and only feed same-field DOFs to avoid cross-component mixing.
        """
        if carrier is None:
            return None
        if hasattr(carrier, "field_name"):
            if getattr(carrier, "field_name", None) != field:
                return None
            return carrier.nodal_values, (getattr(carrier, "_g2l", {}) or {})
        if hasattr(carrier, "field_names"):
            flds = list(getattr(carrier, "field_names", []))
            if field not in flds:
                return None
            return carrier.nodal_values, (getattr(carrier, "_g2l", {}) or {})
        return None

    k_eff = max(1, int(k))

    for fld in field_names:
        sl = field_slices.get(fld)
        if sl is None or sl.size == 0:
            continue
        # Newly active DOFs for this field (intersect)
        new_fld = np.intersect1d(newly_active, sl, assume_unique=False)
        if new_fld.size == 0:
            continue
        # Source DOFs for this field
        src_fld = np.intersect1d(stable_active, sl, assume_unique=False)
        if src_fld.size == 0:
            continue

        # k-NN query *within the same field* to avoid mixing components/fields.
        tree = cKDTree(coords[src_fld])
        d, idx = tree.query(coords[new_fld], k=min(k_eff, src_fld.size))
        d = np.atleast_2d(d)
        idx = np.atleast_2d(idx)
        src_ids = src_fld[idx]

        # weights: inverse distance (avoid div-by-zero)
        w = 1.0 / (d + 1.0e-12)
        w /= np.sum(w, axis=1, keepdims=True)

        cur = field_to_current.get(fld)
        prv = field_to_prev.get(fld)
        cur_data = _carrier_field_data(cur, fld)
        prv_data = _carrier_field_data(prv, fld)
        if cur_data is None or prv_data is None:
            continue
        cur_view, g2l_cur = cur_data
        prv_view, g2l_prv = prv_data

        for row, gd in enumerate(new_fld):
            # pick weighted average from the same carrier arrays via global dof ids
            src = src_ids[row]
            vals = []
            for sgd in src:
                li = g2l_cur.get(int(sgd))
                if li is None:
                    vals.append(0.0)
                else:
                    vals.append(float(cur_view[int(li)]))
            v_new = float(np.dot(w[row], np.asarray(vals, float)))
            li_cur = g2l_cur.get(int(gd))
            li_prv = g2l_prv.get(int(gd))
            if li_cur is not None:
                cur_view[int(li_cur)] = v_new
            if li_prv is not None:
                prv_view[int(li_prv)] = v_new

        if trace:
            print(f"[extend] field={fld}: initialized {int(new_fld.size)} newly-active dofs")


@dataclass
class FSIFormTerms:
    jacobian_form: object
    residual_form: object
    a_vol_f: object
    r_vol_f: object
    a_vol_s: object
    r_vol_s: object
    a_svc: object
    r_svc: object
    a_stab: object
    r_stab: object
    a_reg: object
    r_reg: object
    J_int: object
    R_int: object
    J_int_fluid: object
    R_int_fluid: object
    J_int_solid: object
    R_int_solid: object
    J_int_pen: object
    R_int_pen: object
    J_int_sym_fluid: Optional[object]
    R_int_sym_fluid: Optional[object]
    J_int_sym_solid: Optional[object]
    R_int_sym_solid: Optional[object]
    a_ext: Optional[object]
    r_ext: Optional[object]
    a_supg: Optional[object]
    r_supg: Optional[object]


def build_fsi_eulerian_forms(
    *,
    du_f,
    dp_f,
    du_s,
    ddisp_s,
    test_vel_f,
    test_q_f,
    test_vel_s,
    test_disp_s,
    uf_k,
    pf_k,
    uf_n,
    pf_n,
    us_k,
    us_n,
    disp_k,
    disp_n,
    dx_fluid,
    dx_solid,
    dGamma,
    dG_fluid,
    dG_solid,
    kappa_pos,
    kappa_neg,
    cell_h,
    beta_N,
    rho_f,
    rho_s,
    mu_f,
    mu_s,
    lambda_s,
    dt,
    theta,
    gamma_v,
    gamma_v_s: Optional[object] = None,
    gamma_p,
    gamma_v_grad,
    solid_reg_eps,
    svc_scale=None,
    use_linear_solid: bool = True,
    solid_stvk_paper: bool = False,
    solid_advect_lagged: bool = True,
    s_nitsche_value: float = 1.0,
    interface_form: str = "symmetric",
    fluid_hessian_ghost: bool = False,
    gamma_u_mom: Optional[object] = None,
    w_fluid: Optional[object] = None,
    w_solid: Optional[object] = None,
    dG_fluid_ext: Optional[object] = None,
    dG_solid_ext: Optional[object] = None,
    gamma_v_ext: Optional[object] = None,
    gamma_p_ext: Optional[object] = None,
    gamma_vs_ext: Optional[object] = None,
    gamma_u_ext: Optional[object] = None,
    gamma_u_psi_ext: Optional[object] = None,
    supg_delta0_vs: Optional[object] = None,
    supg_delta0_u: Optional[object] = None,
) -> FSIFormTerms:
    I2 = Identity(2)
    n = FacetNormal()
    interface_form = str(interface_form or "symmetric").lower().strip()
    if interface_form not in {"symmetric", "paper"}:
        raise ValueError("interface_form must be 'symmetric' or 'paper'.")

    def epsilon_f(u):
        return 0.5 * (grad(u) + grad(u).T)

    def traction_fluid_primal(u_vec, p_scal):
        return 2.0 * mu_f * dot(epsilon_f(u_vec), n) - p_scal * n

    def traction_fluid_adjoint(v_vec, q_scal):
        return 2.0 * mu_f * dot(epsilon_f(v_vec), n) + q_scal * n

    def F_of(d):
        return inv(I2 - grad(d))

    def C_of(F):
        return dot(F.T, F)

    def E_of(F):
        return 0.5 * (C_of(F) - I2)

    def S_stvk(E):
        return lambda_s * trace(E) * I2 + Constant(2.0) * mu_s * E

    def sigma_s_linear(d):
        eps = 0.5 * (grad(d) + grad(d).T)
        return Constant(2.0) * mu_s * eps + lambda_s * trace(eps) * I2

    def dsigma_s_linear(delta_d):
        eps = 0.5 * (grad(delta_d) + grad(delta_d).T)
        return Constant(2.0) * mu_s * eps + lambda_s * trace(eps) * I2

    def sigma_s_nonlinear(d):
        if use_linear_solid:
            return sigma_s_linear(d)
        if solid_stvk_paper:
            return sigma_s_stvk_paper(d, mu_s=mu_s, lambda_s=lambda_s)
        F = F_of(d)
        E = E_of(F)
        S = S_stvk(E)
        J = det(F)
        return (Constant(1.0) / J) * dot(dot(F, S), F.T)

    def dsigma_s(d_ref, delta_d):
        if use_linear_solid:
            return dsigma_s_linear(delta_d)
        if solid_stvk_paper:
            return dsigma_s_stvk_paper(d_ref, delta_d, mu_s=mu_s, lambda_s=lambda_s)
        Fk = F_of(d_ref)
        Ek = E_of(Fk)
        Sk = S_stvk(Ek)
        dF = dot(Fk, dot(grad(delta_d), Fk))
        dE = Constant(0.5) * (dot(dF.T, Fk) + dot(Fk.T, dF))
        dS = lambda_s * trace(dE) * I2 + Constant(2.0) * mu_s * dE
        Jk = det(Fk)
        dJ = Jk * trace(dot(grad(delta_d), Fk))
        term = dot(dF, dot(Sk, Fk.T)) + dot(Fk, dot(dS, Fk.T)) + dot(Fk, dot(Sk, dF.T))
        return (Constant(1.0) / Jk) * term - (dJ / Jk) * sigma_s_nonlinear(d_ref)

    def traction_solid_R(d):
        return dot(sigma_s_nonlinear(d), n)

    def traction_solid_L(delta_d, d_ref):
        return dot(dsigma_s(d_ref, delta_d), n)

    def grad_inner_jump(u, v):
        a = dot(jump(grad(u)), n)
        b = dot(jump(grad(v)), n)
        return inner(a, b)

    def hess_inner_jump(u, v):
        a = dot(n, dot(jump(Hessian(u)), n))
        b = dot(n, dot(jump(Hessian(v)), n))
        return inner(a, b)

    def g_v_f(gamma, phi_1, phi_2):
        val = cell_h * grad_inner_jump(phi_1, phi_2)
        if fluid_hessian_ghost:
            val = val + Constant(0.25) * (cell_h**3.0) * hess_inner_jump(phi_1, phi_2)
        return gamma * val

    def g_p(gamma, phi_1, phi_2):
        return gamma * (cell_h**3.0 * grad_inner_jump(phi_1, phi_2))

    def g_v_s(gamma, phi_1, phi_2):
        return gamma * (cell_h**3.0 * grad_inner_jump(phi_1, phi_2))

    def g_disp_s(gamma, phi_1, phi_2):
        return gamma * (cell_h * grad_inner_jump(phi_1, phi_2))

    if svc_scale is None:
        svc_scale = Constant(1.0)

    # Paper parameters distinguish γ_{v_f} and γ_{v_s}. Keep backward
    # compatibility by defaulting the solid velocity stabilization to `gamma_v`.
    gamma_v_s_eff = gamma_v if gamma_v_s is None else gamma_v_s

    wfac_f = Constant(1.0) if w_fluid is None else (Pos(w_fluid) + Neg(w_fluid))
    wfac_s = Constant(1.0) if w_solid is None else (Pos(w_solid) + Neg(w_solid))

    jump_vel_trial = Pos(du_f) - Neg(du_s)
    jump_vel_test = Pos(test_vel_f) - Neg(test_vel_s)
    jump_vel_res = Pos(uf_k) - Neg(us_k)
    jump_test_f = Pos(test_vel_f)
    jump_test_s = Neg(test_vel_s)

    solid_adv_vel = us_n if solid_advect_lagged else us_k

    if interface_form == "paper":
        # Paper Problem 3 (Eq. 7–8): one-sided Nitsche flux using σ_f only.
        # Note: FacetNormal() is oriented from (−) solid to (+) fluid, hence n_f = −n.
        avg_flux_fluid_trial = traction_fluid_primal(Pos(du_f), Pos(dp_f))
        avg_flux_fluid_test = traction_fluid_adjoint(Pos(test_vel_f), Pos(test_q_f))
        avg_flux_fluid_res = traction_fluid_primal(Pos(uf_k), Pos(pf_k))
    else:
        # Symmetric / averaged flux variant (default in existing examples).
        avg_flux_fluid_trial = kappa_pos * traction_fluid_primal(Pos(du_f), Pos(dp_f))
        avg_flux_fluid_test = kappa_pos * traction_fluid_adjoint(Pos(test_vel_f), Pos(test_q_f))
        avg_flux_fluid_res = kappa_pos * traction_fluid_primal(Pos(uf_k), Pos(pf_k))

    # `n` is a single interface normal oriented from (−) solid to (+) fluid.
    # With this convention, outward normals are n_f = −n (fluid) and n_s = +n (solid).
    # Define tractions using the same `n`:  t_f = σ_f n,  t_s = σ_s n.
    #
    # For the volume terms used below (e.g.  μ∫∇u:∇v  and  ∫σ:∇v), integration by parts
    # yields the combined interface contribution
    #   B_Γ(v_f,v_s) = ∫_Γ (σ_f n_f)·v_f + (σ_s n_s)·v_s
    #              = ∫_Γ (−t_f·v_f + t_s·v_s).
    # Under the physical dynamic condition t_f = t_s = t this becomes
    #   B_Γ = −∫_Γ t·(v_f − v_s) = −∫_Γ t·jump(v).
    #
    # A consistent Nitsche term must therefore add
    #   N_Γ(u;v) = +∫_Γ {t(u)}·jump(v),
    # with the *sum* average {t} = κ⁺ t_f + κ⁻ t_s (κ⁺+κ⁻=1), so that {t}=t on the
    # exact interface condition and B_Γ + N_Γ = 0 for arbitrary discontinuous tests.
    avg_flux_solid_trial = None
    avg_flux_solid_test = None
    avg_flux_solid_res = None
    if interface_form != "paper":
        avg_flux_solid_trial = kappa_neg * traction_solid_L(Neg(ddisp_s), Neg(disp_k))
        avg_flux_solid_test = kappa_neg * traction_solid_L(Neg(test_vel_s), Neg(disp_k))
        avg_flux_solid_res = kappa_neg * traction_solid_R(Neg(disp_k))

    s_nitsche = Constant(s_nitsche_value)

    J_int_fluid = (dot(avg_flux_fluid_trial, jump_test_f)) * dGamma - (dot(avg_flux_fluid_trial, jump_test_s)) * dGamma
    R_int_fluid = (dot(avg_flux_fluid_res, jump_test_f)) * dGamma - (dot(avg_flux_fluid_res, jump_test_s)) * dGamma
    if avg_flux_solid_trial is None:
        J_int_solid = Constant(0.0) * dGamma
        R_int_solid = Constant(0.0) * dGamma
    else:
        J_int_solid = (dot(avg_flux_solid_trial, jump_test_f)) * dGamma - (dot(avg_flux_solid_trial, jump_test_s)) * dGamma
        R_int_solid = (dot(avg_flux_solid_res, jump_test_f)) * dGamma - (dot(avg_flux_solid_res, jump_test_s)) * dGamma
    J_int_pen = (beta_N * mu_f / cell_h) * dot(jump_vel_trial, jump_vel_test) * dGamma
    R_int_pen = (beta_N * mu_f / cell_h) * dot(jump_vel_res, jump_vel_test) * dGamma

    J_int = J_int_fluid + J_int_solid + J_int_pen
    R_int = R_int_fluid + R_int_solid + R_int_pen

    J_int_sym_fluid = None
    R_int_sym_fluid = None
    J_int_sym_solid = None
    R_int_sym_solid = None
    if s_nitsche_value != 0.0:
        J_int_sym_fluid = (s_nitsche * dot(avg_flux_fluid_test, jump_vel_trial)) * dGamma
        R_int_sym_fluid = (s_nitsche * dot(avg_flux_fluid_test, jump_vel_res)) * dGamma
        if avg_flux_solid_test is not None:
            J_int_sym_solid = (s_nitsche * dot(avg_flux_solid_test, jump_vel_trial)) * dGamma
            R_int_sym_solid = (s_nitsche * dot(avg_flux_solid_test, jump_vel_res)) * dGamma
            J_int = J_int + J_int_sym_fluid + J_int_sym_solid
            R_int = R_int + R_int_sym_fluid + R_int_sym_solid
        else:
            J_int = J_int + J_int_sym_fluid
            R_int = R_int + R_int_sym_fluid

    # IMPORTANT:
    # Use the *symmetric-gradient* viscous volume form so it is consistent with
    # the traction used in the Nitsche interface fluxes (`traction_fluid_*`).
    # Otherwise, for oblique interfaces (n has both components), the missing
    # transpose correction μ(∇uᵀ·n) shows up as a consistency error and MMS tests
    # can fail or even worsen under refinement.
    a_vol_f = (
        rho_f / dt * dot(du_f, test_vel_f)
        + theta * rho_f * dot(dot(grad(uf_k), du_f), test_vel_f)
        + theta * rho_f * dot(dot(grad(du_f), uf_k), test_vel_f)
        + Constant(2.0) * theta * mu_f * inner(epsilon_f(du_f), epsilon_f(test_vel_f))
        - dp_f * div(test_vel_f)
        + test_q_f * div(du_f)
    ) * dx_fluid

    r_vol_f = (
        rho_f * inner(uf_k - uf_n, test_vel_f) / dt
        + theta * rho_f * dot(dot(grad(uf_k), uf_k), test_vel_f)
        + (Constant(1.0) - theta) * rho_f * dot(dot(grad(uf_n), uf_n), test_vel_f)
        + Constant(2.0) * theta * mu_f * inner(epsilon_f(uf_k), epsilon_f(test_vel_f))
        + Constant(2.0) * (Constant(1.0) - theta) * mu_f * inner(epsilon_f(uf_n), epsilon_f(test_vel_f))
        - pf_k * div(test_vel_f)
        + test_q_f * div(uf_k)
    ) * dx_fluid

    sigma_s_k = sigma_s_nonlinear(disp_k)
    sigma_s_n = sigma_s_nonlinear(disp_n)
    dsigma_s_k = dsigma_s(disp_k, ddisp_s)

    a_vol_s = (
        rho_s * dot(du_s, test_vel_s) / dt
        + theta * inner(dsigma_s_k, grad(test_vel_s))
        + (
            rho_s
            * theta
            * (
                dot(dot(grad(du_s), solid_adv_vel), test_vel_s)
                if solid_advect_lagged
                else (
                    dot(dot(grad(us_k), du_s), test_vel_s)
                    + dot(dot(grad(du_s), us_k), test_vel_s)
                )
            )
        )
    ) * dx_solid
    r_vol_s = (
        rho_s * inner(us_k - us_n, test_vel_s) / dt
        + theta * inner(sigma_s_k, grad(test_vel_s))
        + (Constant(1.0) - theta) * inner(sigma_s_n, grad(test_vel_s))
        + rho_s
        * (
            theta * dot(dot(grad(us_k), solid_adv_vel), test_vel_s)
            + (Constant(1.0) - theta) * dot(dot(grad(us_n), us_n), test_vel_s)
        )
    ) * dx_solid

    a_svc = svc_scale * (
        dot(ddisp_s, test_disp_s) / dt
        - theta * dot(du_s, test_disp_s)
        + (
            theta
            * (
                dot(dot(grad(ddisp_s), solid_adv_vel), test_disp_s)
                if solid_advect_lagged
                else (
                    dot(dot(grad(ddisp_s), us_k), test_disp_s)
                    + dot(dot(grad(disp_k), du_s), test_disp_s)
                )
            )
        )
    ) * dx_solid
    r_svc = svc_scale * (
        inner(disp_k - disp_n, test_disp_s) / dt
        - theta * inner(us_k, test_disp_s)
        - (Constant(1.0) - theta) * inner(us_n, test_disp_s)
        + theta * dot(dot(grad(disp_k), solid_adv_vel), test_disp_s)
        + (Constant(1.0) - theta) * dot(dot(grad(disp_n), us_n), test_disp_s)
    ) * dx_solid

    a_stab = (
        (
            Constant(2.0) * mu_f * wfac_f * g_v_f(gamma_v, du_f, test_vel_f)
            + wfac_f * g_p(gamma_p, dp_f, test_q_f)
        )
        * dG_fluid
        + (
            rho_s * wfac_s * g_v_s(gamma_v_s_eff, du_s, test_vel_s)
            + Constant(2.0) * mu_s * wfac_s * g_disp_s(gamma_v_grad, ddisp_s, test_disp_s)
        )
        * dG_solid
    )
    r_stab = (
        (
            Constant(2.0) * mu_f * wfac_f * g_v_f(gamma_v, uf_k, test_vel_f)
            + wfac_f * g_p(gamma_p, pf_k, test_q_f)
        )
        * dG_fluid
        + (
            rho_s * wfac_s * g_v_s(gamma_v_s_eff, us_k, test_vel_s)
            + Constant(2.0) * mu_s * wfac_s * g_disp_s(gamma_v_grad, disp_k, test_disp_s)
        )
        * dG_solid
    )

    # Optional: stabilize displacement in the solid momentum equation (paper Eq. (13)).
    if gamma_u_mom is not None:
        a_stab = a_stab + (Constant(2.0) * mu_s * wfac_s * g_disp_s(gamma_u_mom, ddisp_s, test_vel_s)) * dG_solid
        r_stab = r_stab + (Constant(2.0) * mu_s * wfac_s * g_disp_s(gamma_u_mom, disp_k, test_vel_s)) * dG_solid

    # Implicit extension ghost penalties (paper Eq. (14)).
    a_ext = None
    r_ext = None
    ext_a_terms = []
    ext_r_terms = []
    if dG_fluid_ext is not None and (gamma_v_ext is not None or gamma_p_ext is not None):
        expr_a_f = None
        expr_r_f = None
        if gamma_v_ext is not None:
            t_a = Constant(2.0) * mu_f * wfac_f * g_v_f(gamma_v_ext, du_f, test_vel_f)
            t_r = Constant(2.0) * mu_f * wfac_f * g_v_f(gamma_v_ext, uf_k, test_vel_f)
            expr_a_f = t_a if expr_a_f is None else expr_a_f + t_a
            expr_r_f = t_r if expr_r_f is None else expr_r_f + t_r
        if gamma_p_ext is not None:
            t_a = wfac_f * g_p(gamma_p_ext, dp_f, test_q_f)
            t_r = wfac_f * g_p(gamma_p_ext, pf_k, test_q_f)
            expr_a_f = t_a if expr_a_f is None else expr_a_f + t_a
            expr_r_f = t_r if expr_r_f is None else expr_r_f + t_r
        if expr_a_f is not None:
            ext_a_terms.append(expr_a_f * dG_fluid_ext)
        if expr_r_f is not None:
            ext_r_terms.append(expr_r_f * dG_fluid_ext)

    if dG_solid_ext is not None and (gamma_vs_ext is not None or gamma_u_ext is not None or gamma_u_psi_ext is not None):
        expr_a_s = None
        expr_r_s = None
        if gamma_vs_ext is not None:
            t_a = rho_s * wfac_s * g_v_s(gamma_vs_ext, du_s, test_vel_s)
            t_r = rho_s * wfac_s * g_v_s(gamma_vs_ext, us_k, test_vel_s)
            expr_a_s = t_a if expr_a_s is None else expr_a_s + t_a
            expr_r_s = t_r if expr_r_s is None else expr_r_s + t_r
        if gamma_u_ext is not None:
            t_a = Constant(2.0) * mu_s * wfac_s * g_disp_s(gamma_u_ext, ddisp_s, test_vel_s)
            t_r = Constant(2.0) * mu_s * wfac_s * g_disp_s(gamma_u_ext, disp_k, test_vel_s)
            expr_a_s = t_a if expr_a_s is None else expr_a_s + t_a
            expr_r_s = t_r if expr_r_s is None else expr_r_s + t_r
        if gamma_u_psi_ext is not None:
            t_a = wfac_s * g_disp_s(gamma_u_psi_ext, ddisp_s, test_disp_s)
            t_r = wfac_s * g_disp_s(gamma_u_psi_ext, disp_k, test_disp_s)
            expr_a_s = t_a if expr_a_s is None else expr_a_s + t_a
            expr_r_s = t_r if expr_r_s is None else expr_r_s + t_r
        if expr_a_s is not None:
            ext_a_terms.append(expr_a_s * dG_solid_ext)
        if expr_r_s is not None:
            ext_r_terms.append(expr_r_s * dG_solid_ext)

    if ext_a_terms:
        a_ext = ext_a_terms[0]
        for t in ext_a_terms[1:]:
            a_ext = a_ext + t
    if ext_r_terms:
        r_ext = ext_r_terms[0]
        for t in ext_r_terms[1:]:
            r_ext = r_ext + t

    # SUPG convection stabilization (paper Sec. 3.3; artificial diffusion form).
    a_supg = None
    r_supg = None
    if supg_delta0_vs is not None or supg_delta0_u is not None:
        vel_mag = (dot(solid_adv_vel, solid_adv_vel) + Constant(1.0e-16)) ** 0.5
        denom = Constant(6.0) * mu_s / rho_s + cell_h * vel_mag + cell_h / dt

        supg_a_terms = []
        supg_r_terms = []
        if supg_delta0_vs is not None:
            delta_vs = supg_delta0_vs * (cell_h**2.0) / denom
            conv_vs = dot(grad(us_k), solid_adv_vel)
            test_stream_vs = dot(grad(test_vel_s), solid_adv_vel)
            supg_r_terms.append(delta_vs * dot(conv_vs, test_stream_vs) * dx_solid)
            supg_a_terms.append(delta_vs * dot(dot(grad(du_s), solid_adv_vel), test_stream_vs) * dx_solid)

        if supg_delta0_u is not None:
            delta_u = supg_delta0_u * (cell_h**2.0) / denom
            conv_u = dot(grad(disp_k), solid_adv_vel)
            test_stream_u = dot(grad(test_disp_s), solid_adv_vel)
            supg_r_terms.append(delta_u * dot(conv_u, test_stream_u) * dx_solid)
            supg_a_terms.append(delta_u * dot(dot(grad(ddisp_s), solid_adv_vel), test_stream_u) * dx_solid)

        if supg_a_terms:
            a_supg = supg_a_terms[0]
            for t in supg_a_terms[1:]:
                a_supg = a_supg + t
        if supg_r_terms:
            r_supg = supg_r_terms[0]
            for t in supg_r_terms[1:]:
                r_supg = r_supg + t

    a_reg = solid_reg_eps * (dot(du_s, test_vel_s) + dot(ddisp_s, test_disp_s)) * dx_solid
    r_reg = solid_reg_eps * (inner(us_k, test_vel_s) + inner(disp_k, test_disp_s)) * dx_solid

    jacobian_form = a_vol_f + J_int + a_vol_s + a_stab + a_svc + a_reg
    residual_form = r_vol_f + R_int + r_vol_s + r_stab + r_svc + r_reg
    if a_ext is not None:
        jacobian_form = jacobian_form + a_ext
    if r_ext is not None:
        residual_form = residual_form + r_ext
    if a_supg is not None:
        jacobian_form = jacobian_form + a_supg
    if r_supg is not None:
        residual_form = residual_form + r_supg

    return FSIFormTerms(
        jacobian_form=jacobian_form,
        residual_form=residual_form,
        a_vol_f=a_vol_f,
        r_vol_f=r_vol_f,
        a_vol_s=a_vol_s,
        r_vol_s=r_vol_s,
        a_svc=a_svc,
        r_svc=r_svc,
        a_stab=a_stab,
        r_stab=r_stab,
        a_reg=a_reg,
        r_reg=r_reg,
        J_int=J_int,
        R_int=R_int,
        J_int_fluid=J_int_fluid,
        R_int_fluid=R_int_fluid,
        J_int_solid=J_int_solid,
        R_int_solid=R_int_solid,
        J_int_pen=J_int_pen,
        R_int_pen=R_int_pen,
        J_int_sym_fluid=J_int_sym_fluid,
        R_int_sym_fluid=R_int_sym_fluid,
        J_int_sym_solid=J_int_sym_solid,
        R_int_sym_solid=R_int_sym_solid,
        a_ext=a_ext,
        r_ext=r_ext,
        a_supg=a_supg,
        r_supg=r_supg,
    )
